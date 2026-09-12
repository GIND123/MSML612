#!/usr/bin/env python3
"""Pull results from Zaratan, refresh the README figures and tables, push to
GitHub and the Hugging Face Hub.

Safe to re-run: the README results block is delimited by markers and replaced
wholesale each time.
"""
import os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = "/scratch/zt1/project/msml612/user/govind02/pcmd"
SSH = ["-o", "ControlMaster=auto",
       "-o", f"ControlPath={Path.home()}/.ssh/cm-zaratan-%r@%h-%p",
       "-o", "ControlPersist=8h"]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def rrun(cmd):
    return subprocess.run(["ssh", *SSH, "zaratan", "bash", "-l", "-c", cmd],
                          capture_output=True, text=True).stdout


def main():
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)

    print("regenerating figures on the cluster ...")
    print(rrun(f"source /etc/profile; module load pytorch/2.0.1 >/dev/null 2>&1; cd {REMOTE}/src && "
               f"RUNS={REMOTE}/runs OUT={REMOTE}/figures python collect.py 2>&1 | tail -3"))

    print("pulling figures ...")
    subprocess.run(["scp", *SSH, "-q", f"zaratan:{REMOTE}/figures/*", str(figs)])

    summary = figs / "summary.md"
    if not summary.exists():
        sys.exit("no summary.md produced")
    tables = summary.read_text()

    readme = HERE / "README.md"
    text = readme.read_text()
    if START in text and END in text:
        text = re.sub(f"{re.escape(START)}.*?{re.escape(END)}",
                      f"{START}\n{tables}\n{END}", text, flags=re.S)
        readme.write_text(text)
        print("README results block refreshed")

    n = len(list(figs.glob("*.png")))
    subprocess.run(["git", "add", "-A"], cwd=HERE)
    r = subprocess.run(["git", "commit", "-q", "-m",
                        f"Update results: {n} figures and regenerated tables"], cwd=HERE)
    if r.returncode == 0:
        subprocess.run(["git", "push", "-q", "origin", "main"], cwd=HERE)
        print("pushed to GitHub")
    else:
        print("nothing new to commit")

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    subprocess.run([sys.executable, str(HERE / "push_to_hf.py")], cwd=HERE)


if __name__ == "__main__":
    main()
