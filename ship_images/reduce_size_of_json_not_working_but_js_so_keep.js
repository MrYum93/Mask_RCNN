#!/usr/bin/env node

const fs = require('fs');
const glob = require('glob');

glob('*.json', {}, (err, files)=>{
  for(i = 0; i < files.length; i++){
    console.log(files[i])
    
    let rawdata = fs.readFileSync(files[i]);
    let data = JSON.parse(rawdata);
    
/*    var data = require('./test.json');
    var data = require(files[i]);
*/
    data.imageHeight = data.imageHeight / 9.6;
    data.imageWidth = data.imageWidth / 9.6;
    //delete data.imageData;
/*
    console.log("all points " + data.shapes[0].points)
    console.log("one deeper " + data.shapes[0].points[0])
    console.log("two points " + data.shapes[0].points[0][0])
    console.log("length if points " + data.shapes[0].points.length)
*/
    for (j = 0; j < data.shapes[0].points.length; j++) {
      for (k = 0; k < data.shapes[0].points[k].length; k++) {
        //console.log(j + " " + data.shapes[0].points[j][k])
        data.shapes[0].points[j][k] /= 9.6;
      }
    }

    data = JSON.stringify(data, null, 2);

    fs.writeFile("resized_" + files[i], data, function(err) {
      if (err) {
        throw err;
      }
      console.log("JSON data is saved.");
    });

  }
})

/*
const fs = require('fs');
const glob = require('glob');

glob('*.json', {}, (err, files)=>{
  for(i = 0; i < files.length; i++){
    console.log(files[i])
    
    let rawdata = fs.readFileSync(files[i]);
    let data = JSON.parse(rawdata);
/*
    console.log(data)
    
    var data = require('./test.json');
    var data = require(files[i]);
*/
/*    data.imageHeight = data.imageHeight / 9.6;
    data.imageWidth = data.imageWidth / 9.6;
/*
    console.log("all points " + data.shapes[0].points)
    console.log("one deeper " + data.shapes[0].points[0])
    console.log("two points " + data.shapes[0].points[0][0])
    console.log("length if points " + data.shapes[0].points.length)
*//*
    for (j = 0; j < data.shapes[0].points.length; j++) {
      for (k = 0; k < data.shapes[0].points[k].length; k++) {
        //console.log(j + " " + data.shapes[0].points[j][k])
        data.shapes[0].points[j][k] /= 9.6;
      }
    }

    data = JSON.stringify(data, null, 2);

    fs.writeFile("resized_" + files[i], data, (err) => {
      if (err) {
        throw err;
      }
      console.log("JSON data is saved.");
    });

  }
})
*/
