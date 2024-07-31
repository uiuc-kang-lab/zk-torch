# GPU Test
`test_gpu_scripts` is a directory that contains scripts to test the GPU on the campus cluster. The scripts are used to test the GPU's performance and to ensure that the GPU is working correctly.

## Workflow
The workflow for testing gpu is in the `.github/workflows/gpu.yml`. The workflow consists of the following steps:
1. uses: actions/checkout@v2: this pulls the repo to the VM hosted by GitHub
2. name: Get a short version of the GIT commit SHA: this gets a 8 chars commit hash to rename our repo later so that we don't mess up different commits on CC when testing
3. name: Copy files over to the cluster: this scp the repo on the VM to CC
4. name: Execute script to enqueue job: this ssh to CC. And it appends necessary dependencies for gpu testing and sends the sbatch job to CC SLURM node to test it

## Notes
- The GitHub Actions may show error messages (e.g., Error: Timed out while waiting for handshake) when CC network is not accessible. This is because the VM hosted by GitHub cannot access the CC network. In this case, the workflow will fail and the user will have to manually cancel the GitHub Actions workflow and re-run it when the network is accessible (i.e., click `Checks` tab and `Re-run all jobs` button under the PR).
