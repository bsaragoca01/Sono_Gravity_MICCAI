import 'package:flutter/material.dart';


class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {

  Map data = {};

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF2F2F2),
      body: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              image: DecorationImage(
                fit: BoxFit.cover,
                image: AssetImage('assets/logo.png'),
              ),
            ),
          ),
          Positioned(
            bottom: 4.0, // Ajuste conforme necessário
            left: 0,
            right: 0,
            child: Center(
              child: ElevatedButton(
                onPressed: () {
                  Navigator.pushNamed(context, '/predict');
                },
                style: ElevatedButton.styleFrom(
                  shape: CircleBorder(
                  side: BorderSide( // Add a black border
                  color: Colors.black,
                  width: 3, // Adjust the border width
                  ),
                  ),
                  elevation: 10,
                  shadowColor: Colors.black,
                  padding: EdgeInsets.all(20), // Adjust padding to increase button size
                  backgroundColor: Color(0xFFbacae7), // Apply #bacae7 as background color
                ),
                child: Image.asset(
                  'assets/probe0.png', // Path to your image
                  width: 60, // Adjust the size of the image
                  height: 60,
              ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
