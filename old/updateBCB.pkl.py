import pandas as pd
import asyncio
import aiohttp
import json
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("process_log.log"),
        logging.StreamHandler()
    ]
)

# Function to interact with the LM Studio server
async def get_code_description(code, semaphore, row_index):
    async with semaphore:
        try:
            # Prepare the input for the LM Studio server
            input_data = {
                "model": "qwen2.5-coder-3b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a senior software engineer. Provide a really short, concise description of the following code: RESPOND WITH DESCRIPTION ONLY, MAXIMUM 7 WORDS, describe main functionality"},
                    {"role": "user", "content": f"{code}"}
                ],
                "temperature": 0.0,
                "max_tokens": -1,
                "stream": False
            }

            # Send a POST request to the LM Studio server
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:1234/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(input_data)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logging.info(f"Row {row_index}: Successfully processed.")
                        return result.get("choices", [{}])[0].get("message", {}).get("content", "No description provided")
                    else:
                        error_message = await response.text()
                        logging.error(f"Row {row_index}: Model error: {error_message}")
                        return f"Error: {error_message}"
        except Exception as e:
            logging.exception(f"Row {row_index}: Exception occurred while processing code: {str(e)}")
            return f"Error: {str(e)}"


async def process_codes(df, start_index, checkpoint_file="BCB_checkpoint.pkl", checkpoint_save_frequency = 100):
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

    for i, row in df.iloc[start_index:].iterrows():
        desc = await get_code_description(row['content'], semaphore, i)
        df.at[i, 'description'] = desc
        logging.info(f"Row {i}: Description saved.")

        # Zapisuj checkpoint co 100 wierszy do checkpointu
        if i % 100 == 0:
            df.to_pickle(checkpoint_file)
            logging.info(f"Checkpoint saved at row {i}.")
    return df


def main():
    checkpoint_file = 'BCB_checkpoint.pkl'
    checkpoint_save_frequency = 100
    final_output_file = 'BCB_updated.pkl'

    try:
        if os.path.exists(checkpoint_file):
            logging.info(f"Loading checkpoint from {checkpoint_file}...")
            df = pd.read_pickle(checkpoint_file)
            start_index = df[df['description'].isna()].index.min()
            logging.info(f"Resuming processing from row {start_index}...")
        else:
            logging.info("Loading data from pickle file...")
            df = pd.read_pickle('BCB.pkl')
            df['description'] = None  # Add a column for descriptions
            start_index = 0
            logging.info("Data loaded successfully.")
    except FileNotFoundError:
        logging.error("Error: File 'BCB.pkl' not found")
        return
    except Exception as e:
        logging.exception(f"Error loading pickle file: {str(e)}")
        return

    try:
        # Run asynchronous processing for the entire dataset
        loop = asyncio.get_event_loop()
        logging.info(f"Processing rows {start_index} to {len(df) - 1}...")
        df = loop.run_until_complete(process_codes(df, start_index, checkpoint_file, checkpoint_save_frequency))

        # Save the final DataFrame to a pickle file
        df.to_pickle(final_output_file)
        logging.info(f"Final DataFrame saved to {final_output_file}.")
        logging.info("Processing completed successfully.")
        logging.info(df[['name', 'description']])
    except Exception as e:
        logging.exception(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()