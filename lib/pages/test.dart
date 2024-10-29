import 'package:flutter/material.dart';
import 'package:web_socket_channel/io.dart'; // Import WebSocket package

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late IOWebSocketChannel channel;
  String receivedMessage = "No messages received yet."; // To store the received message

  @override
  void initState() {
    super.initState();

    // Connect to the WebSocket server
    channel = IOWebSocketChannel.connect('ws://100.68.9.113:12345'); // Adjust the URL as necessary

    // Listen for incoming messages
    channel.stream.listen((message) {
      setState(() {
        receivedMessage = message; // Update the received message
      });
    }, onDone: () {
      print("WebSocket connection closed.");
    }, onError: (error) {
      print("Error in WebSocket connection: $error");
    });

    // Send a "Hello, World!" message when connected
    sendHelloMessage();
  }

  void sendHelloMessage() {
    channel.sink.add("Hello, World!");
    print("Sent: Hello, World!");
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('WebSocket Example'),
          backgroundColor: Color(0xFFbacae7),
        ),
        backgroundColor: Color(0xFFbacae7),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                'Received Message:',
                style: TextStyle(color: Colors.white),
              ),
              SizedBox(height: 20),
              Text(
                receivedMessage,
                style: TextStyle(color: Colors.white),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    channel.sink.close(); // Close the WebSocket connection
    super.dispose();
  }
}
