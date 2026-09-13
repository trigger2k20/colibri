#!/usr/bin/env python3
from pathlib import Path

p = Path("c/coli")
s = p.read_text()
old = '''    if arch=="glm53":
        e=env_for_engine(a,arch)
        cmd=[engine,"--model",os.path.abspath(a.model),"--prompt",prompt,
             "--greedy",str(ngen_for(a,interactive=True,family=family))]
        sys.exit(subprocess.call(cmd,env=e))
'''
new = '''    if arch=="glm53":
        e=env_for_engine(a,arch)
        # GLM53 accepts cache/layer as a positional numeric argument.  Keep the
        # same cap_for_launch contract used by the persistent gateway so an
        # explicit `coli run --cap N` reaches the engine instead of being
        # silently ignored by the one-shot path.
        cap = cap_for_launch(a.cap, e, 0)
        cmd=[engine,str(cap),"--model",os.path.abspath(a.model),"--prompt",prompt,
             "--greedy",str(ngen_for(a,interactive=True,family=family))]
        sys.exit(subprocess.call(cmd,env=e))
'''
if new in s:
    print("GLM53 run cap forwarding already applied")
elif old not in s:
    raise SystemExit("expected GLM53 one-shot launcher block not found")
else:
    p.write_text(s.replace(old, new, 1))
    print("applied GLM53 one-shot --cap forwarding")
