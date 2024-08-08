import sys 

gpu_dir = str(sys.argv[1])

with open("Cargo.toml", "r") as f:
  contents = f.readlines()

for i, c in enumerate(contents):
  if c.strip() == "[dependencies]":
    # add dependencies after [dependencies]
    c = c + 'icicle-cuda-runtime = { path = "' + gpu_dir +'/gpu/icicle/wrappers/rust/icicle-cuda-runtime" }\n'
    c = c + 'icicle-core = { path = "' + gpu_dir +'/gpu/icicle/wrappers/rust/icicle-core", features = ["arkworks"]}\n'
    c = c + 'icicle-bn254 = { path = "' + gpu_dir +'/gpu/icicle/wrappers/rust/icicle-curves/icicle-bn254" , features = ["arkworks", "g2"]}\n'
    contents[i] = c
    break

with open("Cargo.toml", "w") as f:
  contents = "".join(contents)
  f.write(contents)
