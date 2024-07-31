from slack_sdk import WebClient
import sys 

pr_number = sys.argv[1]
token = sys.argv[2]
commit_hash = str(sys.argv[3])[:7]

m = ' starts to run the gpu test for commit: ' + commit_hash + '. '
note = 'If no message is sent after this, the test is either still running or being killed.'
message = 'Event #' + str(pr_number) + m + note


# reference: https://www.datacamp.com/tutorial/how-to-send-slack-messages-with-python
# Set up a WebClient with the Slack OAuth token
client = WebClient(token=token)

# Send a message
client.chat_postMessage(
  channel="zk-torch-test-gpu", 
  text=message, 
  username="SLURM bot"
)
