from slack_sdk import WebClient
import sys 

pr_number = sys.argv[1]
token = sys.argv[2]
commit_hash = str(sys.argv[3])[:7]

with open("zktorch_gh_action.out", "r") as f:
  contents = f.readlines()

commit_str = ' (commit: ' + commit_hash + ')'
message = 'error found when cargo run in event #' + str(pr_number) + commit_str

# Check if log file exists
try:
  with open("zktorch_gh_action.out", "r") as f:
    contents = f.readlines()
  # Check if the log file contains the string Cargo run was successful.
  for line in contents:
    if "Cargo run was successful." in line:
      message = 'Event #' + str(pr_number) + commit_str + ' successfully passed cargo run on CC gpu'
      break
except FileNotFoundError:
  message = 'error found when cargo run in event #' + str(pr_number) + commit_str

# reference: https://www.datacamp.com/tutorial/how-to-send-slack-messages-with-python
# Set up a WebClient with the Slack OAuth token
client = WebClient(token=token)

# Send a message
client.chat_postMessage(
  channel="zk-torch-test-gpu", 
  text=message, 
  username="SLURM bot"
)
