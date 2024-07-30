from slack_sdk import WebClient
import sys 

pr_number = sys.argv[1]
token = sys.argv[2]

message = 'PR #' + str(pr_number) + 'successfully passed the gpu test'
with open("zktorch_gh_action.out", "r") as f:
  contents = f.readlines()

start_scan = False
for c in contents:
  if 'Running `target/debug/zkml_proofs`' in c:
    start_scan = True
  if start_scan and 'error:' in c or "thread 'main' panicked" in c:
    message = 'error found in the gpu test in PR #' + str(pr_number)
    break

# Set up a WebClient with the Slack OAuth token
client = WebClient(token=token)

# Send a message
client.chat_postMessage(
    channel="zk-torch-test-gpu", 
    text=message, 
    username="SLURM bot"
)