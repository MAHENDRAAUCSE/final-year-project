import 'package:flutter/material.dart';
import 'upload_screen.dart';

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Smart STP Predictor")),
      body: Center(
        child: ElevatedButton(
          child: Text("Upload Dataset"),
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => UploadScreen()),
            );
          },
        ),
      ),
    );
  }
}