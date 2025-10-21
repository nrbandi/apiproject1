import asyncio
from prefect import get_client  # This is the main API client

async def fetch_deployment_details():
    """
    Connects to the Prefect API and fetches deployment details.
    """
    print("--- Sub-objective 3: API Access ---")
    print("Connecting to Prefect Cloud API to fetch deployment info...")

    # get_client() automatically finds your active workspace (nrbandi/default)
    async with get_client() as client:
        try:
            # 1. Use the API to read all deployments in the workspace
            deployments = await client.read_deployments()

            print(f"\nFound {len(deployments)} deployments:")

            for i, dep in enumerate(deployments):
                print(f"\n  --- Deployment {i+1} ---")
                # Displaying 4 application details as required
                print(f"  1. Name: {dep.name}")
                print(f"  2. Status: {dep.status}")

                # We can also get the ID of the flow this deployment runs
                flow = await client.read_flow(dep.flow_id)
                print(f"  3. Flow Name: {flow.name}")
                print(f"  4. Flow ID: {dep.flow_id}")

            print("\n--- API Access Successful ---")

        except Exception as e:
            print(f"An error occurred while fetching data from the API: {e}")

# This is the standard way to run an async 'main' function
if __name__ == "__main__":
    asyncio.run(fetch_deployment_details())