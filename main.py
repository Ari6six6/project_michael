"""
michael — air-gapped, event-sourced, AI-native control loop.

This file is a thin compatibility shim. All logic lives in the michael/ package.
"""
# Re-export everything the test suite needs via `import main as m`
import typer  # noqa: F401 — tests reference m.typer

import michael.globals as _G

from michael.globals import (  # noqa: F401
    STATE_DIR,
    GLOBAL_CONFIG_PATH,
    GLOBAL_EVENTS_PATH,
    STATE_FILE_PATH,
    PROJECTS_DIR,
    REPL_HISTORY_PATH,
    console,
    err,
    MAX_FILE_BYTES_INLINE,
    MAX_TOTAL_BYTES_INLINE,
    SKIP_DIRS,
    AUTO_EXEC_TOOLS,
    MichaelError,
    _GOD_MODE_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
)

from michael.config import (  # noqa: F401
    ModelProfile,
    VpsConfig,
    SandboxConfig,
    Config,
    make_stub_config,
    CONFIG_HELP,
)

from michael.project import (  # noqa: F401
    Project,
    slugify,
    list_projects,
    create_project,
    get_active_slug,
    set_active_slug,
    get_active_project,
    require_active_project,
    _last_seq,
    _append,
    append_event,
    iter_events,
    replay_global,
)

from michael.backends import (  # noqa: F401
    _ssh_argv,
    _ssh_preflight,
    VastClient,
    llm_client,
    chat_stream,
    _ping_vllm,
    _require_endpoint,
    _build_vllm_cmd,
    _restart_vllm_on_gpu,
    _ensure_tunnel,
    _close_tunnel,
    SandboxBackend,
    LocalPodmanBackend,
    RemotePodmanBackend,
    make_backend,
)

from michael.utils import (  # noqa: F401
    _is_text,
    filesystem_snapshot,
    _prompt_history_lines,
    _action_log_lines,
    build_protocol,
    build_header,
)

from michael.tools import (  # noqa: F401
    TOOLS,
    _resolve_in_project,
    _summary_for,
    _format_proc_result,
    _stage_ignore,
    _stage_project,
    _file_hashes,
    _diff_hashes,
    _check_expected,
    _run_verify,
    _apply_in_staging,
    _save_trash,
    _sync_to_real,
    _list_trash,
    _undo_one,
    _search_memory,
    execute_tool,
    _format_delta,
    _format_review,
    PendingChanges,
    _snapshot_file,
    _restore_file,
    execute_with_staging,
    commit_pending,
    _render_for_confirmation,
    _edit_args,
    confirm_tool_call,
    dispatch_tool_call,
)

from michael.agent import (  # noqa: F401
    _run_agent_loop,
)

from michael.cli import (  # noqa: F401
    app,
    cmd_init,
    cmd_show,
    cmd_new,
    cmd_use,
    cmd_current,
    cmd_config,
    cmd_up,
    cmd_down,
    cmd_status,
    cmd_ask,
    cmd_run,
    cmd_log,
    cmd_undo,
    cmd_sandbox,
    cmd_ssh_test,
    REPL_COMMANDS,
    _config_is_unset,
    MichaelCompleter,
    repl,
    _opt_value,
    dispatch_repl,
    main,
)


if __name__ == "__main__":
    main()
