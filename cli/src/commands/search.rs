// Copyright 2025 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use clap_complete::ArgValueCandidates;
use clap_complete::ArgValueCompleter;
use itertools::Itertools as _;
use jj_lib::backend::CommitId;
use jj_lib::commit::Commit;
use jj_lib::graph::GraphEdge;
use jj_lib::graph::GraphEdgeType;
use jj_lib::graph::TopoGroupedGraphIterator;
use jj_lib::graph::reverse_graph;
use jj_lib::repo::Repo as _;
use jj_lib::revset::RevsetEvaluationError;
use jj_lib::revset::RevsetExpression;
use jj_lib::revset::RevsetFilterPredicate;
use jj_lib::revset::RevsetIteratorExt as _;
use jj_lib::str_util::StringExpression;
use jj_lib::str_util::StringPattern;
use pollster::FutureExt as _;
use tracing::instrument;

use crate::cli_util::CommandHelper;
use crate::cli_util::LogContentFormat;
use crate::cli_util::RevisionArg;
use crate::cli_util::format_template;
use crate::command_error::CommandError;
use crate::command_error::user_error;
use crate::complete;
use crate::diff_util::DiffFormatArgs;
use crate::formatter::FormatterExt as _;
use crate::graphlog::GraphStyle;
use crate::graphlog::get_graphlog;
use crate::templater::TemplateRenderer;
use crate::ui::Ui;

/// Search for patterns in commit diffs
///
/// Finds commits where the diff contains the specified pattern(s). By default,
/// searches for any line in the diff that matches the pattern (like `git log -G`).
///
/// Search modes can be changed using flags. Each flag changes the mode for
/// subsequent patterns:
///
/// - `--mentioned` (default): Pattern appears in any changed line (added or removed)
/// - `--added`: Pattern appears only in added lines
/// - `--removed`: Pattern appears only in removed lines
///
/// Multiple patterns are combined with AND logic (all must match).
///
/// Patterns support type prefixes like `regex:`, `glob:`, `exact:`, `substring:`.
/// The default is `substring` matching.
///
/// Examples:
///
/// ```
/// # Find commits mentioning "TODO"
/// jj search TODO
///
/// # Find commits where "foo" was added and "bar" was removed
/// jj search --added foo --removed bar
///
/// # Use regex pattern
/// jj search 'regex:TODO.*FIXME'
///
/// # Limit to specific revisions
/// jj search -r 'trunk()..@' foo
///
/// # Restrict to specific files
/// jj search foo -- '*.rs'
/// ```
#[derive(clap::Args, Clone, Debug)]
#[command(verbatim_doc_comment)]
pub(crate) struct SearchArgs {
    /// Which revisions to search
    #[arg(long, short, default_value = "::", value_name = "REVSETS")]
    #[arg(add = ArgValueCompleter::new(complete::revset_expression_all))]
    revisions: Vec<RevisionArg>,

    /// Limit number of revisions to show
    #[arg(long, short = 'n')]
    limit: Option<usize>,

    /// Show revisions in the opposite order (older revisions first)
    #[arg(long)]
    reversed: bool,

    /// Don't show the graph, show a flat list of revisions
    #[arg(long)]
    no_graph: bool,

    /// Render each revision using the given template
    ///
    /// Run `jj log -T` to list the built-in templates.
    #[arg(long, short = 'T')]
    #[arg(add = ArgValueCandidates::new(complete::template_aliases))]
    template: Option<String>,

    /// Show patch
    #[arg(long, short = 'p')]
    patch: bool,

    #[command(flatten)]
    diff_format: DiffFormatArgs,

    /// Search patterns with mode flags (--mentioned, --added, --removed)
    ///
    /// Mode flags change the interpretation of subsequent patterns.
    /// Use -- to separate patterns from file restrictions.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, value_name = "PATTERNS")]
    raw_patterns: Vec<String>,
}

/// The mode for matching a pattern in the diff.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchMode {
    /// Pattern appears in any changed line (added or removed).
    /// This is like `git log -G`.
    #[default]
    Mentioned,
    /// Pattern appears only in added lines.
    Added,
    /// Pattern appears only in removed lines.
    Removed,
    // Future: Changed mode for occurrence count changes (like `git log -S`)
}

/// A search pattern with its associated mode.
#[derive(Clone, Debug)]
pub struct SearchPattern {
    pub mode: SearchMode,
    pub pattern: StringPattern,
}

/// Parsed arguments from the raw pattern list.
struct ParsedPatterns {
    patterns: Vec<SearchPattern>,
    filesets: Vec<String>,
}

/// Parses the raw pattern arguments into structured patterns and filesets.
///
/// Handles mode-switching flags (--mentioned, --added, --removed) and
/// the -- separator for filesets.
fn parse_raw_patterns(raw: &[String]) -> Result<ParsedPatterns, CommandError> {
    let mut patterns = vec![];
    let mut filesets = vec![];
    let mut current_mode = SearchMode::Mentioned;
    let mut after_separator = false;

    for arg in raw {
        if arg == "--" {
            after_separator = true;
            continue;
        }
        if after_separator {
            filesets.push(arg.clone());
            continue;
        }
        match arg.as_str() {
            "--mentioned" => current_mode = SearchMode::Mentioned,
            "--added" => current_mode = SearchMode::Added,
            "--removed" => current_mode = SearchMode::Removed,
            "--changed" => {
                // Reserved for future use
                return Err(user_error("--changed mode is not yet implemented"));
            }
            value => {
                let pattern = parse_string_pattern(value)?;
                patterns.push(SearchPattern {
                    mode: current_mode,
                    pattern,
                });
            }
        }
    }
    Ok(ParsedPatterns { patterns, filesets })
}

/// Parses a string pattern with optional kind prefix (e.g., "regex:foo").
fn parse_string_pattern(src: &str) -> Result<StringPattern, CommandError> {
    let (maybe_kind, pat) = src
        .split_once(':')
        .map_or((None, src), |(kind, pat)| (Some(kind), pat));

    if let Some(kind) = maybe_kind {
        StringPattern::from_str_kind(pat, kind)
            .map_err(|err| user_error(format!("Invalid pattern: {err}")))
    } else {
        // Default to substring matching
        Ok(StringPattern::substring(pat))
    }
}

#[instrument(skip_all)]
pub(crate) fn cmd_search(
    ui: &mut Ui,
    command: &CommandHelper,
    args: &SearchArgs,
) -> Result<(), CommandError> {
    let workspace_command = command.workspace_helper(ui)?;
    let settings = workspace_command.settings();

    // Parse the raw patterns
    let parsed = parse_raw_patterns(&args.raw_patterns)?;

    if parsed.patterns.is_empty() {
        return Err(user_error(
            "No search patterns provided. Usage: jj search <pattern>",
        ));
    }

    // For MVP, we only support --mentioned mode
    // Check if any pattern uses unsupported modes
    for pattern in &parsed.patterns {
        if pattern.mode != SearchMode::Mentioned {
            return Err(user_error(format!(
                "{:?} mode is not yet implemented. Only --mentioned is supported.",
                pattern.mode
            )));
        }
    }

    // Parse filesets if provided
    let fileset_expression = workspace_command.parse_file_patterns(ui, &parsed.filesets)?;

    // Build the revset expression
    let mut revset_expression = workspace_command.parse_union_revsets(ui, &args.revisions)?;

    // Apply diff_contains filter for each pattern (AND logic)
    for search_pattern in &parsed.patterns {
        let text = StringExpression::pattern(search_pattern.pattern.clone());
        let predicate = RevsetFilterPredicate::DiffContains {
            text,
            files: fileset_expression.clone(),
        };
        revset_expression.intersect_with(&RevsetExpression::filter(predicate));
    }

    let revset = revset_expression.evaluate()?;

    let repo = workspace_command.repo();
    let matcher = fileset_expression.to_matcher();

    let store = repo.store();
    let diff_renderer = workspace_command.diff_renderer_for_log(&args.diff_format, args.patch)?;
    let graph_style = GraphStyle::from_settings(settings)?;

    let use_elided_nodes = settings.get_bool("ui.log-synthetic-elided-nodes")?;
    let with_content_format = LogContentFormat::new(ui, settings)?;

    let template: TemplateRenderer<Commit>;
    let node_template: TemplateRenderer<Option<Commit>>;
    {
        let language = workspace_command.commit_template_language();
        let template_string = match &args.template {
            Some(value) => value.clone(),
            None => settings.get_string("templates.log")?,
        };
        template = workspace_command
            .parse_template(ui, &language, &template_string)?
            .labeled(["log", "commit"]);
        node_template = workspace_command
            .parse_template(ui, &language, &settings.get_string("templates.log_node")?)?
            .labeled(["log", "commit", "node"]);
    }

    {
        ui.request_pager();
        let mut formatter = ui.stdout_formatter();
        let formatter = formatter.as_mut();

        if !args.no_graph {
            let mut raw_output = formatter.raw()?;
            let mut graph = get_graphlog(graph_style, raw_output.as_mut());
            let iter: Box<dyn Iterator<Item = _>> = {
                let forward_iter = TopoGroupedGraphIterator::new(revset.iter_graph(), |id| id);
                let forward_iter = forward_iter.take(args.limit.unwrap_or(usize::MAX));
                if args.reversed {
                    Box::new(reverse_graph(forward_iter, |id| id)?.into_iter().map(Ok))
                } else {
                    Box::new(forward_iter)
                }
            };
            for node in iter {
                let (commit_id, edges) = node?;

                let mut graphlog_edges = vec![];
                let mut missing_edge_id = None;
                let mut elided_targets = vec![];
                for edge in edges {
                    match edge.edge_type {
                        GraphEdgeType::Missing => {
                            missing_edge_id = Some(edge.target);
                        }
                        GraphEdgeType::Direct => {
                            graphlog_edges.push(GraphEdge::direct((edge.target, false)));
                        }
                        GraphEdgeType::Indirect => {
                            if use_elided_nodes {
                                elided_targets.push(edge.target.clone());
                                graphlog_edges.push(GraphEdge::direct((edge.target, true)));
                            } else {
                                graphlog_edges.push(GraphEdge::indirect((edge.target, false)));
                            }
                        }
                    }
                }
                if let Some(missing_edge_id) = missing_edge_id {
                    graphlog_edges.push(GraphEdge::missing((missing_edge_id, false)));
                }
                let mut buffer = vec![];
                let key = (commit_id, false);
                let commit = store.get_commit(&key.0)?;
                let within_graph =
                    with_content_format.sub_width(graph.width(&key, &graphlog_edges));
                within_graph.write(ui.new_formatter(&mut buffer).as_mut(), |formatter| {
                    template.format(&commit, formatter)
                })?;
                if let Some(renderer) = &diff_renderer {
                    let mut formatter = ui.new_formatter(&mut buffer);
                    renderer
                        .show_patch(
                            ui,
                            formatter.as_mut(),
                            &commit,
                            matcher.as_ref(),
                            within_graph.width(),
                        )
                        .block_on()?;
                }

                let commit = Some(commit);
                let node_symbol = format_template(ui, &commit, &node_template);
                graph.add_node(
                    &key,
                    &graphlog_edges,
                    &node_symbol,
                    &String::from_utf8_lossy(&buffer),
                )?;

                for elided_target in elided_targets {
                    let elided_key = (elided_target, true);
                    let real_key = (elided_key.0.clone(), false);
                    let edges = [GraphEdge::direct(real_key)];
                    let mut buffer = vec![];
                    let within_graph =
                        with_content_format.sub_width(graph.width(&elided_key, &edges));
                    within_graph.write(ui.new_formatter(&mut buffer).as_mut(), |formatter| {
                        writeln!(formatter.labeled("elided"), "(elided revisions)")
                    })?;
                    let node_symbol = format_template(ui, &None, &node_template);
                    graph.add_node(
                        &elided_key,
                        &edges,
                        &node_symbol,
                        &String::from_utf8_lossy(&buffer),
                    )?;
                }
            }
        } else {
            let iter: Box<dyn Iterator<Item = Result<CommitId, RevsetEvaluationError>>> = {
                let forward_iter = revset.iter().take(args.limit.unwrap_or(usize::MAX));
                if args.reversed {
                    let entries: Vec<_> = forward_iter.try_collect()?;
                    Box::new(entries.into_iter().rev().map(Ok))
                } else {
                    Box::new(forward_iter)
                }
            };
            for commit_or_error in iter.commits(store) {
                let commit = commit_or_error?;
                with_content_format
                    .write(formatter, |formatter| template.format(&commit, formatter))?;
                if let Some(renderer) = &diff_renderer {
                    let width = ui.term_width();
                    renderer
                        .show_patch(ui, formatter, &commit, matcher.as_ref(), width)
                        .block_on()?;
                }
            }
        }
    }

    Ok(())
}
