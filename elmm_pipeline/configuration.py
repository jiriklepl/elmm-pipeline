from pathlib import Path
import re
import shutil


def data_lines(path):
    return [s for line in Path(path).read_text().splitlines()
            if (s := re.split(r"[!#]", line)[0].strip())]


def prepare_config(source, destination, *, steps=None, advection=None):
    source, destination = Path(source).resolve(), Path(destination)
    if source.is_file():
        source = source.parent
    if destination.resolve() == source or source in destination.resolve().parents:
        raise ValueError("Configuration snapshot must be outside the input configuration directory")
    if not (source / "main.conf").is_file():
        raise ValueError(f"Configuration needs main.conf: {source}")
    if destination.exists():
        raise ValueError(f"Configuration snapshot already exists: {destination}")
    # Copy every asset, including input/ and relative profiles. Dereference symlinks
    # so the snapshot can be moved independently of the source configuration.
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    original_advection = int(data_lines(destination / "main.conf")[0])
    overrides = {}
    if advection is not None:
        path = destination / "main.conf"
        lines = path.read_text().splitlines(keepends=True)
        index = next(i for i, s in enumerate(lines) if re.split(r"[!#]", s)[0].strip())
        lines[index] = f"{advection} ! explicit pipeline --advection override\n"
        path.write_text("".join(lines))
        overrides["advection_method"] = {"original": original_advection, "effective": advection}
    if steps is not None:
        if steps <= 0:
            raise ValueError("--steps must be positive")
        path = destination / "time_stepping.conf"
        text = path.read_text()
        # Preserve offsets while masking comments, including inline and multiline
        # assignments. Never insert a duplicate of an existing active limit.
        active = re.sub(r"[!#][^\n]*", lambda m: " " * len(m[0]), text)
        assignments = list(re.finditer(r"(?i)\bmax_number_of_time_steps\s*=", active))
        if len(assignments) > 1:
            raise ValueError("Duplicate active max_number_of_time_steps assignments")
        if assignments:
            match = re.match(r"\s*[+-]?\d+", active[assignments[0].end():])
            if not match:
                raise ValueError("Cannot parse existing max_number_of_time_steps value")
            start = assignments[0].end()
            text = text[:start] + f" {steps}" + text[start + match.end():]
        else:
            block = re.search(r"(?i)time_stepping\s*\(", active)
            if not block:
                raise ValueError("Cannot locate time_stepping(...) to apply --steps")
            text = text[:block.end()] + f"\n  max_number_of_time_steps = {steps}," + text[block.end():]
        path.write_text(text)
        overrides["max_number_of_time_steps"] = steps
    method = int(data_lines(destination / "main.conf")[0])
    cmd_file = destination / "cmd.conf"
    cmd_text = cmd_file.read_text().splitlines()[0] if cmd_file.exists() and cmd_file.read_text().splitlines() else ""
    discretization = re.search(r"(?i)\bdiscretization_order\s*=\s*(\d+)", cmd_text.split("!")[0])
    if discretization and int(discretization[1]) == 4:
        method = 4  # ELMM applies this override after reading main.conf.
    text = (destination / "time_stepping.conf").read_text()
    text = "\n".join(re.split(r"[!#]", s)[0] for s in text.splitlines())
    maximum = re.search(r"(?i)max_number_of_time_steps\s*=\s*(\d+)", text)
    return {"source": str(source), "overrides": overrides, "advection_method": method,
            "expected_max_steps": int(maximum[1]) if maximum else None,
            "runtime_arguments": ["output_dir='output/'"]}
