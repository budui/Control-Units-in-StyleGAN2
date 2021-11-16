## How to set up environment?

### From sketch

First, install python library:

```bash
# CUDA 11
conda install pytorch torchvision ignite cudatoolkit=11.1 -c pytorch -c nvidia
# CUDA 10.1
conda install -y pytorch ignite torchvision cudatoolkit=10.1 -c pytorch

# with pip
pip install opencv-python ipython loguru fire tqdm lmdb omegaconf numpy scipy matplotlib pandas
# with conda
conda install -c conda-forge ipython loguru fire tqdm python-lmdb omegaconf

# opencv must be installed with pip
pip install opencv-python
```

Then, install useful tools:

```bash
conda install -c conda-forge ipdb tensorboard
```

Finally, update gcc version if you use `centos/tlinux`:

```bash
# install scl and gcc tools
yum install tlinux-release-scl
yum install devtoolset-7-gcc-c++.x86_64
# enable high version gcc with `scl`
scl enable devtoolset-7 bash
# replace c++ with g++ to avoid PyTorch warnings.
CCPATH=$(which c++); mv $CCPATH "$CCPATH".backup; ln -s $(which g++) $CCPATH
```

## Useful tools:

Check&Kill zombie processes if multi-process tools exit unexpectedly.

```bash
# check zombie processes
ps aux | grep train_generator.py | grep -v grep | awk '{print $2}'
# kill them
ps aux | grep train_generator.py | grep -v grep | awk '{print $2}' | xargs kill -9
```

Show process tree: `ps auxf`

See where a process hang: `strace -p <PID>`

```text
ipdb> from IPython import embed
ipdb> embed() # drop into an IPython session.
        # Any variables you define or modify here
        # will not affect program execution
```