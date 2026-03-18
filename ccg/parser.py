"""
Scope-tree parser: syntax-aware chunking with Tree-sitter and ghost text.
Produces nodes at file, class, function, and optional block level with content hashes.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from tree_sitter_languages import get_language, get_parser
except ImportError:
    get_language = None
    get_parser = None


# Default extensions to include when discovering files
DEFAULT_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp"}

# Directories to ignore when discovering files (Python, JS/TS, and general project cruft)
IGNORED_DIRS = frozenset({
    # VCS and config
    ".git",
    ".svn",
    ".hg",
    # Python
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".eggs",
    "eggs",
    ".tox",
    ".nox",
    # Node / JS / TS
    "node_modules",
    ".next",
    ".nuxt",
    "out",
    ".cache",
    "coverage",
    ".nyc_output",
    ".parcel-cache",
    ".turbo",
    ".vite",
    "storybook-static",
    # Rust
    "target",
    # Go
    # (no standard dir name; vendor is sometimes used)
    # General / IDE
    ".idea",
    ".vscode",
    ".cursor",
    "vendor",  # often third-party
})

# File name patterns to skip (exact suffix or substring in basename)
IGNORED_FILE_PATTERNS = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".chunk.js",
)

# Tree-sitter language name by extension
EXT_TO_LANG: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
}


class ScopeNode:
    """A single node in the scope tree (file, class, function, or block)."""

    __slots__ = (
        "id",
        "type",
        "path",
        "name",
        "class_name",
        "content",
        "ghost_text",
        "signature",
        "start_line",
        "end_line",
        "content_hash",
        "calls",  # list of (name, node_id or None) for CALLS edges
        "uses_types",  # list of type names for USES_TYPE edges
    )

    def __init__(
        self,
        id: str,
        type: str,
        path: str,
        name: str,
        class_name: Optional[str],
        content: str,
        ghost_text: str,
        signature: str,
        start_line: int,
        end_line: int,
        content_hash: str,
        calls: Optional[List[Tuple[str, Optional[str]]]] = None,
        uses_types: Optional[List[str]] = None,
    ):
        self.id = id
        self.type = type
        self.path = path
        self.name = name
        self.class_name = class_name
        self.content = content
        self.ghost_text = ghost_text
        self.signature = signature
        self.start_line = start_line
        self.end_line = end_line
        self.content_hash = content_hash
        self.calls = calls or []
        self.uses_types = uses_types or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "path": self.path,
            "name": self.name,
            "class_name": self.class_name,
            "content": self.content,
            "ghost_text": self.ghost_text,
            "signature": self.signature,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content_hash": self.content_hash,
            "calls": self.calls,
            "uses_types": self.uses_types,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScopeNode":
        return cls(
            id=d["id"],
            type=d["type"],
            path=d["path"],
            name=d["name"],
            class_name=d.get("class_name"),
            content=d["content"],
            ghost_text=d["ghost_text"],
            signature=d.get("signature", ""),
            start_line=d["start_line"],
            end_line=d["end_line"],
            content_hash=d["content_hash"],
            calls=[tuple(x) for x in d.get("calls", [])],
            uses_types=d.get("uses_types", []),
        )


def _content_hash(content: str) -> str:
    normalized = content.strip().replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _serialize_ast_node(node: Any, source: bytes, include_text: bool = True, max_text_len: int = 200) -> Dict[str, Any]:
    """Turn a tree-sitter Node into a JSON-serializable dict."""
    d: Dict[str, Any] = {
        "type": node.type,
        "start_line": node.start_point[0] + 1,
        "start_column": node.start_point[1],
        "end_line": node.end_point[0] + 1,
        "end_column": node.end_point[1],
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
    }
    if include_text:
        text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        d["text"] = text if len(text) <= max_text_len else text[:max_text_len] + "..."
    if node.child_count > 0:
        d["children"] = [_serialize_ast_node(child, source, include_text, max_text_len) for child in node.children]
    return d


def _make_id(path: str, kind: str, name: str, line: int) -> str:
    safe_path = path.replace(" ", "_")
    return f"{safe_path}::{kind}::{name}::{line}"


_python_parser_unavailable_logged = False


def _parse_python(file_path: str, content: str, rel_path: str) -> Tuple[List[ScopeNode], Optional[Dict[str, Any]]]:
    """Returns (scope_nodes, ast_dict for JSON export). ast_dict is None if parser unavailable."""
    global _python_parser_unavailable_logged
    if get_parser is None or get_language is None:
        if not _python_parser_unavailable_logged:
            logger.warning("tree-sitter not available; Python scope-tree disabled. Using file-level nodes only.")
            _python_parser_unavailable_logged = True
        return [], None
    try:
        parser = get_parser("python")
    except Exception as e:
        if not _python_parser_unavailable_logged:
            logger.warning("Python parser not available (%s). Using file-level nodes only.", e)
            _python_parser_unavailable_logged = True
        return [], None

    source = content.encode("utf-8")
    tree = parser.parse(source)
    root = tree.root_node
    nodes: List[ScopeNode] = []
    ast_dict: Optional[Dict[str, Any]] = {"path": rel_path, "language": "python", "ast": _serialize_ast_node(root, source, include_text=True, max_text_len=300)}

    # File-level summary (first 500 chars or first 20 lines)
    lines = content.splitlines()
    summary_lines = lines[:20]
    summary = "\n".join(summary_lines)
    if len(summary) > 500:
        summary = summary[:500] + "\n..."
    file_ghost = f"path: {rel_path} | file summary"
    file_id = _make_id(rel_path, "file", Path(file_path).name, 1)
    nodes.append(
        ScopeNode(
            id=file_id,
            type="file",
            path=rel_path,
            name=Path(file_path).name,
            class_name=None,
            content=summary,
            ghost_text=file_ghost + "\n" + summary,
            signature="",
            start_line=1,
            end_line=min(20, len(lines)),
            content_hash=_content_hash(summary),
        )
    )

    def extract_docstring(node: Any) -> str:
        # Python: first string in body is often docstring
        body = node.child_by_field_name("body")
        if body and body.child_count > 0:
            first = body.child(0)
            if first.type == "expression_statement":
                first = first.child(0)
            if first.type == "string":
                return _node_text(first, source).strip().strip('"\'')
        return ""

    def get_signature(node: Any, name: str) -> str:
        return _node_text(node, source).split("\n")[0]

    def collect_calls_and_types(node: Any) -> Tuple[List[Tuple[str, Optional[str]]], List[str]]:
        calls: List[Tuple[str, Optional[str]]] = []
        types: List[str] = []
        # Walk AST for call expressions and type annotations
        def walk(n: Any) -> None:
            if n.type == "call":
                # (identifier or attribute) for the called name
                fn = n.child_by_field_name("function")
                if fn:
                    call_name = _node_text(fn, source).strip()
                    calls.append((call_name, None))
            if n.type == "type":
                types.append(_node_text(n, source).strip())
            for child in n.children:
                walk(child)
        walk(node)
        return calls, types

    def visit(node: Any, class_name: Optional[str] = None) -> None:
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                cname = _node_text(name_node, source).strip()
                sig = get_signature(node, cname)
                doc = extract_docstring(node)
                body = node.child_by_field_name("body")
                body_text = _node_text(body, source) if body else ""
                class_content = sig + "\n" + doc + "\n" + body_text[:500] if body_text else sig + "\n" + doc
                if len(class_content) > 2000:
                    class_content = class_content[:2000] + "\n..."
                ghost = f"path: {rel_path} | class: {cname} | {sig}"
                nid = _make_id(rel_path, "class", cname, node.start_point[0] + 1)
                nodes.append(
                    ScopeNode(
                        id=nid,
                        type="class",
                        path=rel_path,
                        name=cname,
                        class_name=None,
                        content=class_content,
                        ghost_text=ghost + "\n" + class_content,
                        signature=sig,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content_hash=_content_hash(class_content),
                    )
                )
                for child in node.children:
                    visit(child, cname)
            return
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                fname = _node_text(name_node, source).strip()
                sig = get_signature(node, fname)
                doc = extract_docstring(node)
                body_text = _node_text(node, source)
                ghost = f"path: {rel_path} | class: {class_name or ''} | " if class_name else f"path: {rel_path} | "
                ghost += f"def {fname}"
                full_ghost = ghost + "\n" + body_text
                nid = _make_id(rel_path, "function", fname, node.start_point[0] + 1)
                calls_list, types_list = collect_calls_and_types(node)
                nodes.append(
                    ScopeNode(
                        id=nid,
                        type="function",
                        path=rel_path,
                        name=fname,
                        class_name=class_name,
                        content=body_text,
                        ghost_text=full_ghost,
                        signature=sig,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content_hash=_content_hash(body_text),
                        calls=calls_list,
                        uses_types=types_list,
                    )
                )
            return
        for child in node.children:
            visit(child, class_name)

    visit(root)
    return nodes, ast_dict


def _get_parser_for_lang(lang: str) -> Any:
    """Return tree-sitter parser for language, or None if unavailable."""
    if get_parser is None:
        return None
    try:
        return get_parser(lang)
    except Exception:
        return None


def _js_ts_collect_calls(node: Any, source: bytes) -> List[Tuple[str, Optional[str]]]:
    """Collect call names from JS/TS AST (call_expression: function can be identifier or member_expression)."""
    calls: List[Tuple[str, Optional[str]]] = []

    def walk(n: Any) -> None:
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            if fn:
                if fn.type == "identifier":
                    calls.append((_node_text(fn, source).strip(), None))
                elif fn.type == "member_expression":
                    # a.b.c -> take "c" (property) for simple name resolution
                    prop = fn.child_by_field_name("property")
                    if prop:
                        calls.append((_node_text(prop, source).strip(), None))
                    else:
                        calls.append((_node_text(fn, source).strip(), None))
                else:
                    calls.append((_node_text(fn, source).strip(), None))
        for child in n.children:
            walk(child)

    walk(node)
    return calls


def _parse_javascript(file_path: str, content: str, rel_path: str, lang_id: str = "javascript") -> Tuple[List[ScopeNode], Optional[Dict[str, Any]]]:
    """Parse JS or JSX: class_declaration, function_declaration, method_definition. Returns (nodes, ast_dict)."""
    parser = _get_parser_for_lang(lang_id)
    if parser is None:
        return [], None
    source = content.encode("utf-8")
    tree = parser.parse(source)
    root = tree.root_node
    nodes: List[ScopeNode] = []
    ast_dict: Dict[str, Any] = {"path": rel_path, "language": lang_id, "ast": _serialize_ast_node(root, source, include_text=True, max_text_len=300)}

    lines = content.splitlines()
    summary = "\n".join(lines[:20])
    if len(summary) > 500:
        summary = summary[:500] + "\n..."
    file_id = _make_id(rel_path, "file", Path(file_path).name, 1)
    nodes.append(
        ScopeNode(
            id=file_id,
            type="file",
            path=rel_path,
            name=Path(file_path).name,
            class_name=None,
            content=summary,
            ghost_text=f"path: {rel_path} | file\n" + summary,
            signature="",
            start_line=1,
            end_line=min(20, len(lines)),
            content_hash=_content_hash(summary),
        )
    )

    def get_name(node: Any) -> Optional[str]:
        name_node = node.child_by_field_name("name")
        if name_node:
            return _node_text(name_node, source).strip()
        return None

    def get_signature_line(node: Any) -> str:
        return _node_text(node, source).split("\n")[0]

    def visit(node: Any, class_name: Optional[str] = None) -> None:
        if node.type == "class_declaration":
            cname = get_name(node)
            if cname:
                sig = get_signature_line(node)
                body = node.child_by_field_name("body")
                body_text = _node_text(body, source) if body else ""
                class_content = sig + "\n" + body_text[:500] if body_text else sig
                if len(class_content) > 2000:
                    class_content = class_content[:2000] + "\n..."
                nid = _make_id(rel_path, "class", cname, node.start_point[0] + 1)
                nodes.append(
                    ScopeNode(
                        id=nid,
                        type="class",
                        path=rel_path,
                        name=cname,
                        class_name=None,
                        content=class_content,
                        ghost_text=f"path: {rel_path} | class: {cname} | {sig}\n" + class_content,
                        signature=sig,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content_hash=_content_hash(class_content),
                    )
                )
                for child in node.children:
                    visit(child, cname)
            return
        if node.type in ("function_declaration", "generator_function_declaration"):
            fname = get_name(node)
            if fname:
                sig = get_signature_line(node)
                body_text = _node_text(node, source)
                calls_list = _js_ts_collect_calls(node, source)
                ghost = f"path: {rel_path} | class: {class_name or ''} | " if class_name else f"path: {rel_path} | "
                ghost += f"function {fname}"
                nid = _make_id(rel_path, "function", fname, node.start_point[0] + 1)
                nodes.append(
                    ScopeNode(
                        id=nid,
                        type="function",
                        path=rel_path,
                        name=fname,
                        class_name=class_name,
                        content=body_text,
                        ghost_text=ghost + "\n" + body_text,
                        signature=sig,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content_hash=_content_hash(body_text),
                        calls=calls_list,
                        uses_types=[],
                    )
                )
            return
        if node.type == "method_definition":
            name_node = node.child_by_field_name("name")
            fname = _node_text(name_node, source).strip() if name_node else None
            if not fname or fname in ("constructor", "get", "set"):
                fname = fname or "constructor"
            sig = get_signature_line(node)
            body_text = _node_text(node, source)
            calls_list = _js_ts_collect_calls(node, source)
            ghost = f"path: {rel_path} | class: {class_name or ''} | {fname}"
            nid = _make_id(rel_path, "function", fname, node.start_point[0] + 1)
            nodes.append(
                ScopeNode(
                    id=nid,
                    type="function",
                    path=rel_path,
                    name=fname,
                    class_name=class_name,
                    content=body_text,
                    ghost_text=ghost + "\n" + body_text,
                    signature=sig,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    content_hash=_content_hash(body_text),
                    calls=calls_list,
                    uses_types=[],
                )
            )
            return
        for child in node.children:
            visit(child, class_name)

    visit(root)
    return nodes, ast_dict


def _parse_java(file_path: str, content: str, rel_path: str) -> Tuple[List[ScopeNode], Optional[Dict[str, Any]]]:
    """Parse Java: class_declaration, method_declaration, interface_declaration. Collect method_invocation for CALLS."""
    parser = _get_parser_for_lang("java")
    if parser is None:
        return [], None
    source = content.encode("utf-8")
    tree = parser.parse(source)
    root = tree.root_node
    nodes: List[ScopeNode] = []
    ast_dict: Dict[str, Any] = {"path": rel_path, "language": "java", "ast": _serialize_ast_node(root, source, include_text=True, max_text_len=300)}

    lines = content.splitlines()
    summary = "\n".join(lines[:20])
    if len(summary) > 500:
        summary = summary[:500] + "\n..."
    file_id = _make_id(rel_path, "file", Path(file_path).name, 1)
    nodes.append(
        ScopeNode(
            id=file_id,
            type="file",
            path=rel_path,
            name=Path(file_path).name,
            class_name=None,
            content=summary,
            ghost_text=f"path: {rel_path} | file\n" + summary,
            signature="",
            start_line=1,
            end_line=min(20, len(lines)),
            content_hash=_content_hash(summary),
        )
    )

    def get_name(node: Any) -> Optional[str]:
        name_node = node.child_by_field_name("name")
        if name_node:
            return _node_text(name_node, source).strip()
        return None

    def get_signature_line(node: Any) -> str:
        return _node_text(node, source).split("\n")[0]

    def collect_java_calls(node: Any) -> List[Tuple[str, Optional[str]]]:
        calls: List[Tuple[str, Optional[str]]] = []

        def walk(n: Any) -> None:
            if n.type == "method_invocation":
                name_node = n.child_by_field_name("name")
                if name_node:
                    calls.append((_node_text(name_node, source).strip(), None))
            for child in n.children:
                walk(child)

        walk(node)
        return calls

    def visit(node: Any, class_name: Optional[str] = None) -> None:
        if node.type == "class_declaration":
            cname = get_name(node)
            if cname:
                sig = get_signature_line(node)
                body = node.child_by_field_name("body")
                body_text = _node_text(body, source) if body else ""
                class_content = sig + "\n" + body_text[:500] if body_text else sig
                if len(class_content) > 2000:
                    class_content = class_content[:2000] + "\n..."
                nid = _make_id(rel_path, "class", cname, node.start_point[0] + 1)
                nodes.append(
                    ScopeNode(
                        id=nid,
                        type="class",
                        path=rel_path,
                        name=cname,
                        class_name=None,
                        content=class_content,
                        ghost_text=f"path: {rel_path} | class: {cname} | {sig}\n" + class_content,
                        signature=sig,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content_hash=_content_hash(class_content),
                    )
                )
                for child in node.children:
                    visit(child, cname)
            return
        if node.type == "interface_declaration":
            cname = get_name(node)
            if cname:
                sig = get_signature_line(node)
                body = node.child_by_field_name("body")
                body_text = _node_text(body, source) if body else ""
                nid = _make_id(rel_path, "class", cname, node.start_point[0] + 1)
                nodes.append(
                    ScopeNode(
                        id=nid,
                        type="class",
                        path=rel_path,
                        name=cname,
                        class_name=None,
                        content=sig + "\n" + body_text[:500],
                        ghost_text=f"path: {rel_path} | interface: {cname} | {sig}",
                        signature=sig,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content_hash=_content_hash(sig + body_text),
                    )
                )
                for child in node.children:
                    visit(child, cname)
            return
        if node.type == "method_declaration":
            fname = get_name(node)
            if fname:
                sig = get_signature_line(node)
                body_text = _node_text(node, source)
                calls_list = collect_java_calls(node)
                nid = _make_id(rel_path, "function", fname, node.start_point[0] + 1)
                nodes.append(
                    ScopeNode(
                        id=nid,
                        type="function",
                        path=rel_path,
                        name=fname,
                        class_name=class_name,
                        content=body_text,
                        ghost_text=f"path: {rel_path} | class: {class_name or ''} | {fname}\n" + body_text,
                        signature=sig,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content_hash=_content_hash(body_text),
                        calls=calls_list,
                        uses_types=[],
                    )
                )
            return
        if node.type == "constructor_declaration":
            fname = get_name(node) or "constructor"
            sig = get_signature_line(node)
            body_text = _node_text(node, source)
            calls_list = collect_java_calls(node)
            nid = _make_id(rel_path, "function", fname, node.start_point[0] + 1)
            nodes.append(
                ScopeNode(
                    id=nid,
                    type="function",
                    path=rel_path,
                    name=fname,
                    class_name=class_name,
                    content=body_text,
                    ghost_text=f"path: {rel_path} | class: {class_name or ''} | constructor\n" + body_text,
                    signature=sig,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    content_hash=_content_hash(body_text),
                    calls=calls_list,
                    uses_types=[],
                )
            )
            return
        for child in node.children:
            visit(child, class_name)

    visit(root)
    return nodes, ast_dict


def _parse_file(file_path: str, content: str, rel_path: str, extensions: Optional[set] = None) -> Tuple[List[ScopeNode], Optional[Dict[str, Any]]]:
    ext = Path(file_path).suffix.lower()
    if extensions is not None and ext not in extensions:
        return [], None
    lang = EXT_TO_LANG.get(ext)
    if lang == "python":
        nodes, ast_dict = _parse_python(file_path, content, rel_path)
        if nodes:
            return nodes, ast_dict
        # Python scope-tree failed (e.g. tree-sitter not loaded); fall back to file-level node
        summary = content[:1000] + ("..." if len(content) > 1000 else "")
        ghost = f"path: {rel_path} | file"
        nid = _make_id(rel_path, "file", Path(file_path).name, 1)
        return [
            ScopeNode(
                id=nid,
                type="file",
                path=rel_path,
                name=Path(file_path).name,
                class_name=None,
                content=summary,
                ghost_text=ghost + "\n" + summary,
                signature="",
                start_line=1,
                end_line=content.count("\n") + 1,
                content_hash=_content_hash(summary),
            )
        ], None
    if lang == "javascript":
        nodes, ast_dict = _parse_javascript(file_path, content, rel_path, lang_id="javascript")
        if nodes:
            return nodes, ast_dict
    if lang == "typescript":
        nodes, ast_dict = _parse_javascript(file_path, content, rel_path, lang_id="typescript")
        if nodes:
            return nodes, ast_dict
    if lang == "tsx":
        nodes, ast_dict = _parse_javascript(file_path, content, rel_path, lang_id="tsx")
        if nodes:
            return nodes, ast_dict
    if lang == "java":
        nodes, ast_dict = _parse_java(file_path, content, rel_path)
        if nodes:
            return nodes, ast_dict
    # Fallback: single file-level node for other supported languages (no AST stored)
    if lang:
        summary = content[:1000] + ("..." if len(content) > 1000 else "")
        ghost = f"path: {rel_path} | file"
        nid = _make_id(rel_path, "file", Path(file_path).name, 1)
        return [
            ScopeNode(
                id=nid,
                type="file",
                path=rel_path,
                name=Path(file_path).name,
                class_name=None,
                content=summary,
                ghost_text=ghost + "\n" + summary,
                signature="",
                start_line=1,
                end_line=content.count("\n") + 1,
                content_hash=_content_hash(summary),
            )
        ], None
    return [], None


def _should_ignore(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    # Skip if any path component is an ignored directory
    for p in parts:
        if p in IGNORED_DIRS:
            return True
        if p.endswith(".egg-info"):
            return True
        if p.startswith(".") and p not in (".github", ".gitignore", ".env.example"):
            return True
    # Skip minified / built assets
    name = path.name
    for pattern in IGNORED_FILE_PATTERNS:
        if pattern in name:
            return True
    return False


def discover_files(root_path: str | Path, extensions: Optional[set] = None) -> List[Path]:
    root = Path(root_path).resolve()
    if not root.is_dir():
        logger.warning("discover_files: %s is not a directory", root)
        return []
    exts = extensions or DEFAULT_EXTENSIONS
    files: List[Path] = []
    skipped = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.relative_to(root).parts[0] == ".git":
                skipped += 1
                continue
        except ValueError:
            continue
        if path.suffix.lower() not in exts:
            skipped += 1
            continue
        if _should_ignore(path, root):
            skipped += 1
            continue
        files.append(path)
    logger.debug("discover_files: %d files kept, %d skipped (root=%s, exts=%s)", len(files), skipped, root, exts)
    return files


def parse_files(
    root_path: str | Path,
    file_paths: List[Path],
    extensions: Optional[set] = None,
    collect_ast: Optional[Dict[str, Any]] = None,
) -> List[ScopeNode]:
    """
    Parse only the given files under root_path. If collect_ast is not None, fill it with path -> ast_dict (Python only).
    """
    root = Path(root_path).resolve()
    all_nodes: List[ScopeNode] = []
    for path in file_paths:
        try:
            rel = path.resolve().relative_to(root).as_posix()
            content = path.read_text(encoding="utf-8", errors="replace")
            nodes, ast_dict = _parse_file(str(path), content, rel, extensions)
            all_nodes.extend(nodes)
            if collect_ast is not None and ast_dict is not None:
                collect_ast[rel] = ast_dict
        except Exception as e:
            logger.debug("Skip %s: %s", path, e)
    return all_nodes


def parse_codebase(
    root_path: str | Path,
    extensions: Optional[set] = None,
    collect_ast: Optional[Dict[str, Any]] = None,
) -> List[ScopeNode]:
    """
    Discover files under root_path, parse each with scope-tree, return all nodes.
    If collect_ast is not None, it is filled with path -> ast_dict for each file that produced an AST (Python only).
    """
    root = Path(root_path).resolve()
    files = discover_files(root, extensions)
    return parse_files(root, files, extensions, collect_ast=collect_ast)
