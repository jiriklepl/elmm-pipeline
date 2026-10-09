"""Export capture-safe numerical leaf source through the public compiler contract.

This is extraction only: every available leaf is offered independently. The
compiler chooses scopes, numerical legality, effects, transfers and placement.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from pathlib import Path

from . import structural as s
from .gpu_calls import _helper


def export_analysis_sources(project, output_dir: Path, *, source_files) -> Path:
    """Publish configured analysis text while retaining original edit identities."""
    output_dir=Path(output_dir).resolve()
    sources=list(dict.fromkeys(Path(path).resolve(strict=True) for path in source_files))
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    entries,dependencies=[],{}
    output_dir.mkdir(parents=True,exist_ok=True)
    commands=[]
    for original in sources:
        parsed=project.parse(original)
        path=output_dir/(hashlib.sha256(str(original).encode()).hexdigest()[:16]+".f90")
        path.write_text(parsed.preprocessed_text)
        for source,_line in parsed.mapping:
            if source is not None and source!=original and str(source) not in hashes:
                dependencies[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        entries.append({"source":str(original),"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                        "line_map":[line if source==original and line>0 else None for source,line in parsed.mapping]})
        commands.append(parsed.preprocess_command)
    document={"schema_version":1,"source_inputs":hashes,"preserves_source_order":True,
              "configuration":{"defines":project.defines,"preprocessor":shlex.split(os.environ.get("FC","gfortran")),
                               "commands":commands},
              "dependencies":dependencies,"entries":sorted(entries,key=lambda item:item["source"])}
    path=output_dir/"analysis-sources.json"
    path.write_text(json.dumps(document,indent=2)+"\n")
    return path


def export_numerical_sources(project, entry: str, output_dir: Path, *, source_files) -> dict:
    """Export bounded direct-call leaf captures; leave rejected leaves native.

    Sources must be the same configured source inputs later supplied to the
    compiler. Preprocessing/edit provenance is a separate integration concern;
    this package never substitutes analysis text for an original source file.
    """
    output_dir=Path(output_dir).resolve()
    sources=list(dict.fromkeys(Path(p).resolve(strict=True) for p in source_files))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for path in sources:
        project.parse(path)
    matches=[]
    for parsed,routine in list(project.procedures.values()):
        try:
            own,module,_dummies,_insertion=s.original_scope(parsed,routine,project,declarations_only=True)
        except s.Unsupported:
            continue
        qualified=(module+"::"+own.name).lower()
        if qualified==entry.lower() or "::" not in entry and own.name.lower()==entry.lower():
            matches.append((parsed,routine,own))
    if len(matches)!=1:
        raise s.Unsupported("source-package entry is unavailable or ambiguous")
    entries,rejected,visited=[],[],set()

    def visit(parsed,routine,own,depth):
        identity=(str(parsed.path),own.name.lower())
        if identity in visited:
            return
        if depth>=8 or len(visited)>=64:
            rejected.append({"procedure":own.name,"reason":"bounded extraction closure exhausted"})
            return
        visited.add(identity)
        execution=next((n for n in s.content(routine) if s.kind(n)=="Execution_Part"),None)
        for call in (n for n in s.walk(execution) if s.kind(n)=="Call_Stmt"):
            name=call.items[0]
            target=own.resolve(str(name)) if s.kind(name)=="Name" else None
            resolved=project.procedures.get(id(target))
            if resolved is None:
                rejected.append({"procedure":str(name),"reason":"no direct module procedure source"})
                continue
            child_parsed,child=resolved
            try:
                child_scope,module,_dummies,_insertion=s.original_scope(child_parsed,child,project,declarations_only=True)
            except s.Unsupported as error:
                rejected.append({"procedure":str(name),"reason":str(error)})
                continue
            procedure=(module+"::"+child_scope.name).lower()
            child_execution=next((n for n in s.content(child) if s.kind(n)=="Execution_Part"),None)
            if any(s.kind(n)=="Call_Stmt" for n in s.walk(child_execution)):
                visit(child_parsed,child,child_scope,depth+1)
                continue
            if any(item["procedure"]==procedure for item in entries):
                continue
            try:
                region,child_scope,_bindings=_helper(call,own)
                if str(child_parsed.path) not in hashes:
                    raise s.Unsupported("numerical helper source is not a supplied compiler input")
                header=next(n for n in s.content(child) if s.kind(n)=="Subroutine_Stmt")
                formals={str(n).lower() for n in getattr(header.items[2],"items",())}
                parameters=[]
                roots={}
                for metadata in region["symbols"]:
                    name=metadata["name"].lower()
                    symbol=child_scope.resolve(name)
                    if symbol is None:
                        raise s.Unsupported("normalized capture has no original source binding")
                    if symbol.scope is child_scope:
                        root="argument::"+name if name in formals else procedure+"::"+name
                    else:
                        root=symbol.scope.name.lower()+"::"+symbol.name.lower()
                    roots[name]=root
                    parameters.append({"name":name,"resource":root,
                                       **({"physical_origin":[0]*symbol.rank} if symbol.rank else {})})
                for argument,actual in zip(region["args"][len(parameters):],region["actual_args"][len(parameters):],strict=True):
                    bound=re.fullmatch(r"lbound\((\w+), (\d+)\)",actual,re.IGNORECASE)
                    if bound is None or bound[1].lower() not in roots:
                        raise s.Unsupported("normalized helper requires an unsupported extra parameter")
                    parameters.append({"name":argument.lower(),"resource":roots[bound[1].lower()],
                                       "lower_bound_dimension":int(bound[2])})
                output_dir.mkdir(parents=True,exist_ok=True)
                path=output_dir/(region["id"]+".f90")
                path.write_text(region["module_source"])
                entries.append({"procedure":procedure,"source_sha256":hashes[str(child_parsed.path)],
                                "path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                "entry":(region["generated_module"]+"::"+region["entry"]).lower(),
                                "normalization":"whole_storage_rebased_v1","participation":"serial_coordinator",
                                "capture_safe":True,"preserves_source_order":True,"parameters":parameters})
            except s.Unsupported as error:
                rejected.append({"procedure":procedure,"reason":str(error)})

    visit(*matches[0],0)
    output_dir.mkdir(parents=True,exist_ok=True)
    package={"schema_version":1,"source_inputs":hashes,"entries":sorted(entries,key=lambda item:item["procedure"])}
    package_path=output_dir/"numerical-sources.json"
    package_path.write_text(json.dumps(package,indent=2)+"\n")
    report={"package":str(package_path),"entries":len(entries),"rejected":rejected,
            "decisions":"source/capture extraction only; compiler owns legality, scopes and execution"}
    (output_dir/"extraction.json").write_text(json.dumps(report,indent=2)+"\n")
    return report
