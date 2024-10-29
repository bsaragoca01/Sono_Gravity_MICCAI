const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

//Create WebSocket server on port 8080. The user must change it if desired
const wss = new WebSocket.Server({ port: 8080 });

const frameDir = 'path_to_frames_from_a_US_video'; // Replace with your path
const frames = fs.readdirSync(frameDir).filter(file => file.endsWith('.jpg')); //Get all image files

wss.on('connection', (ws) => {
  console.log('Client connected');
  let currentFrame = 0;
  //Stream frames at 14.667 frames per second 
  const frameRate = 1000 / 14.667; // 14.667 FPS 

  const frameInterval = setInterval(() => {
    //Read the current frame
    const framePath = path.join(frameDir, frames[currentFrame]);
    fs.readFile(framePath, (err, data) => {
      if (err) {
        console.error('Error reading frame:', err);
        return;
      }
      //Send the current frame to the connected client, which is the App in this case
      ws.send(data);

      //Move to the next frame. In this specific example, considering the purpose, the frames sequence is looped back to the beginning if necessary
      currentFrame = (currentFrame + 1) % frames.length;
    });
  }, frameRate);
  //Stop streaming when client disconnects
  ws.on('close', () => {
    console.log('Client disconnected');
    clearInterval(frameInterval);
  });
});
console.log('WebSocket server is running on ws://localhost:8080');
