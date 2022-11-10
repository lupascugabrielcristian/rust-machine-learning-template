Ref: https://www.freecodecamp.org/news/how-to-build-a-machine-learning-model-in-rust

### Rust install
Ref: https://www.digitalocean.com/community/tutorials/install-rust-on-ubuntu-linux
curl https://sh.rustup.rs -sSf | sh
source ~/.cargo/env

### Add linfa dependencies, csv, ndarray
In Cargo.toml add
```
linfa = "0.6.0"
linfa-trees = "0.6.0"
linfa-datasets = { version = "0.6.0", features = ["iris"] }
csv = "1.1"
ndarray = "0.15.6"
```
