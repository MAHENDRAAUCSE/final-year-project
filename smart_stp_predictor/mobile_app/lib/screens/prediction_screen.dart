import 'package:flutter/material.dart';

class PredictionScreen extends StatelessWidget {
  final List<dynamic> predictions;

  PredictionScreen({required this.predictions});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("BOD Predictions")),
      body: Column(
        children: [
          Container(
            padding: EdgeInsets.all(16.0),
            color: Colors.blue[50],
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Next ${predictions.length} Days Forecast",
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue[900],
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  "BOD (mg/L)",
                  style: TextStyle(fontSize: 14, color: Colors.grey[700]),
                ),
              ],
            ),
          ),
          Expanded(
            child: ListView.builder(
              itemCount: predictions.length,
              itemBuilder: (context, index) {
                double value = (predictions[index] is num)
                    ? (predictions[index] as num).toDouble()
                    : double.parse(predictions[index].toString());
                
                return Card(
                  margin: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  child: ListTile(
                    leading: CircleAvatar(
                      child: Text("${index + 1}"),
                      backgroundColor: Colors.blue,
                      foregroundColor: Colors.white,
                    ),
                    title: Text("Day ${index + 1}"),
                    trailing: Text(
                      "${value.toStringAsFixed(2)} mg/L",
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.blue[700],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}