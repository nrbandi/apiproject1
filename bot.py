import discord
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()  # This loads your .env file

# Read the token safely from the environment
# This line MUST come AFTER load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# This is the full, unique name of your deployment
# Format: "Flow Name/Deployment Name"
DEPLOYMENT_TO_RUN = "ML Training and Evaluation Pipeline/ml-pipeline-deployment"
# ---------------------

# These "intents" are permissions the bot needs
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True # You enabled this in the portal
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    """Tells us when the bot has successfully logged in."""
    print(f'Bot has logged in as {client.user}')
    # Check if the token was loaded correctly
    if not DISCORD_BOT_TOKEN:
        print("!!! ERROR: DISCORD_BOT_TOKEN was not loaded. Check your .env file.")

@client.event
async def on_message(message):
    """Fires on every message the bot can see."""
    
    # Prevent the bot from replying to itself
    if message.author == client.user:
        return

    # Check if the message is the command we want
    if message.content.startswith('!run-pipeline'):
        print("Received command, triggering Prefect pipeline...")
        
        # Send a confirmation reply *before* doing the work
        await message.channel.send(
            f"✅ Got it, {message.author.mention}! "
            f"Contacting the Prefect API to start `{DEPLOYMENT_TO_RUN}`..."
        )

        # --- This is the corrected API call ---
        try:
            # We must import the client *after* loading the env
            from prefect import get_client

            # Connect to the Prefect API
            async with get_client() as prefect_client:
                print(f"Looking up deployment: {DEPLOYMENT_TO_RUN}")
                
                # Step 1: Find the deployment by its full name
                deployment = await prefect_client.read_deployment_by_name(
                    name=DEPLOYMENT_TO_RUN
                )
                
                print(f"Found deployment ID: {deployment.id}. Creating flow run...")

                # Step 2: Create a flow run from that deployment's ID
                await prefect_client.create_flow_run_from_deployment(
                    deployment_id=deployment.id
                )
            
            # Send a success message
            print("Successfully triggered deployment.")
            await message.channel.send(
                f"🚀 Success! The pipeline is now scheduled. "
                f"Your worker in Terminal 2 will pick it up."
            )

        except Exception as e:
            print(f"Error triggering pipeline: {e}")
            await message.channel.send(
                f"❌ Uh oh, {message.author.mention}. I failed to trigger the pipeline via the API. Error: {e}"
            )
        # ---------------------------------

# --- Run the bot ---
try:
    print("Starting bot...")
    # Add a check *before* running
    if not DISCORD_BOT_TOKEN:
        raise ValueError("Error running bot: DISCORD_BOT_TOKEN is None. Check your .env file and variable name.")
    
    client.run(DISCORD_BOT_TOKEN)
except Exception as e:
    print(e)