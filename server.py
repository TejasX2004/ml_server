import asyncio
import websockets
import json
import os
from server import process_frame  # Import the frame processing function
from dotenv import load_dotenv

# Load environment variables from .env file, useful for local development
load_dotenv()

# Get the port from the environment, defaulting to 3000 if not set
port = int(os.getenv("PORT", 3000))

async def handle_connection(websocket, path):
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data['type'] == 'frame':
                    processed_image_data, stream_id = process_frame(data)
                    await websocket.send(json.dumps({
                        'type': 'processed_frame',
                        'frame': processed_image_data,
                        'streamId': stream_id
                    }))
                    print('Image received, processed, and sent back to client.')
                elif data['type'] == 'info':
                    print('Info:', data['message'])
                elif data['type'] == 'error':
                    print('Error:', data['message'], data['details'])
            except json.JSONDecodeError:
                print('Received a non-JSON message:', message)
            except Exception as e:
                print('Error processing message:', e)
    except websockets.exceptions.ConnectionClosed as e:
        print(f'Connection closed: {e}')

async def main():
    print(f'Starting WebSocket server on port {port}...')
    server = await websockets.serve(handle_connection, "0.0.0.0", port)
    print(f'WebSocket server started on ws://0.0.0.0:{port}')
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
