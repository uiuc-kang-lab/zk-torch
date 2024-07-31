# GPU Test
`test_gpu_scripts` is a directory that contains scripts to test the GPU on the Illinois Campus Cluster. The scripts are used to ensure that the GPU feature is working correctly.

## Workflow
The workflow for testing gpu is in the `.github/workflows/gpu.yml`. The workflow consists of the following steps:
1. uses: actions/checkout@v2: this pulls the repo to the VM hosted by GitHub
2. name: Copy files over to the cluster: this scp the repo on the VM to CC
3. name: Sleep for 1 minute: this is to let CC prepare for the next step
4. name: Execute script to enqueue job: this ssh to CC. And it appends necessary dependencies for gpu testing and sends the sbatch job to CC SLURM node to test it

## Notification
After step 4. above, the user will receive a notice in our slack channel `#zk-torch-test-gpu` that the job has been submitted. Once the job is done, the user will receive another notice in the same channel.

## Notes
- The GitHub Actions may show error messages (e.g., Error: Timed out while waiting for handshake) sometimes. This is because the VM hosted by GitHub cannot access the Campus Cluster network. In this case, the workflow will fail and the user will have to re-run it when the Campus Cluster network is accessible (i.e., click `Checks` tab and `Re-run all jobs` button under the PR).
