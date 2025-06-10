### PyOomph Installation on Mac Silicon (Rosetta Mode)
PyOomph is a computational fluid dynamics framework that requires x86_64 architecture on Mac Silicon. Here's the successful installation process:

1. **Create x86_64 Conda Environment** (in Rosetta terminal):
   ```bash
   # Explicitly set x86_64 architecture
   CONDA_SUBDIR=osx-64 conda create -n py-oomph-conda python=3.11 -y
   ```

2. **Configure Environment for x86_64**:
   ```bash
   conda activate py-oomph-conda
   conda config --env --set subdir osx-64
   ```

3. **Install Dependencies**:
   ```bash
   # Install x86_64 MPI (critical - homebrew's MPI is arm64)
   conda install -c conda-forge openmpi mpi4py -y
   
   # Install pyoomph (automatically installs mkl 2021.4.0)
   python -m pip install pyoomph
   ```

4. **Verify Installation**:
   ```bash
   python -m pyoomph check all
   ```

**Key Points**:
- Must use Rosetta terminal (check with `arch` - should show i386)
- mkl 2021.4.0 is automatically installed (required for Mac Silicon)
- Use conda's x86_64 openmpi to avoid architecture conflicts
- All solver checks should pass: superlu, pardiso, scipy eigen, compiler
