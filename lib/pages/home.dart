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
    return Scaffold( // The Home page design is defined here
      backgroundColor: Color(0xFFF2F2F2),
      body: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              image: DecorationImage(
                fit: BoxFit.cover,
                image: AssetImage('assets/logo.png'), //App logo
              ),
            ),
          ),
          Positioned(
            bottom: 4.0,
            left: 0,
            right: 0,
            child: Center(
              child: ElevatedButton( //Button that when pressed the page is redirected to the Predictions Page.
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
                  padding: EdgeInsets.all(20),
                  backgroundColor: Color(0xFFbacae7),
                ),
                child: Image.asset(
                  'assets/probe0.png', //Image displayed in the button
                  width: 60,
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
