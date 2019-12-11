var threshold = function(image,bandname)
{
  var minInternal = 0.1;
  var maxSection = 100;
  image = image.select([bandname]);
  
  // var histogram1 = image.reduceRegion({"reducer": ee.Reducer.fixedHistogram(0, 1, 10), "tileScale" : 4, "scale" : 1000}).get(bandname);
  // print(histogram1);
  // compute the histogram
  var histogram = image.reduceRegion({"reducer": ee.Reducer.histogram(maxSection, minInternal), "tileScale" : 4}).get(bandname);
  // print("histogram : ", histogram);
  var bucketMin = ee.Number(ee.Array(ee.Dictionary(histogram).get('bucketMin')));
  // print("bucketMin : ", bucketMin);
  var bucketWidth = ee.Number(ee.Array(ee.Dictionary(histogram).get('bucketWidth')));
  // print("bucketWidth : ", bucketWidth);
  var bucketMeans = ee.Array(ee.Dictionary(histogram).get('bucketMeans'));
  // print("bucketMeans : ", bucketMeans);
  histogram = ee.Array(ee.Dictionary(histogram).get('histogram'));
  // print("histogram : ", histogram);
  var internalcount = histogram.length().get([0]);
  // print("internalcount : ", internalcount);
    
  // the count of pixels
  var array_count = image.multiply(0.0).add(1.0).reduceRegion({"reducer": ee.Reducer.sum(), "tileScale" : 4}).get(bandname);
  array_count = ee.Number(array_count);
  // print(array_count);
  
  // compute the accumulation
  var indices = ee.List.sequence(1, histogram.length().get([0]));
  var accumulation = indices.map(function(n) {
    // var value = ee.Number(n);
      var aCounts = histogram.slice(0, 0, n);
      var aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    // print(value);
    return aCount;
  });
  // print("accumulation : ", accumulation);
  
  // compute the entropy
  var propArray = histogram.divide(ee.List.repeat(array_count,internalcount));
  var entropy = propArray.add(propArray.eq(0)).log().multiply(propArray);
  entropy = ee.Array(ee.List.repeat(0, internalcount)).subtract(entropy);
  // print("entropy : ", entropy);
  var entropyaccumulation = indices.map(function(n) {
    // var value = ee.Number(n);
      var aCounts = entropy.slice(0, 0, n);
      var aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    // print(value);
    return aCount;
  });
  // print("entropyaccumulation : ", entropyaccumulation);
  var H = entropy.reduce(ee.Reducer.sum(), [0]).get([0]);
  // print(H);
  
  
  var subentList = ee.List.sequence(0, ee.Number(histogram.length().get([0])).subtract(2)).map(function(n) {
    var index = ee.Number(n);

    // compute Pt
    var pt = ee.Number(accumulation.get(index)).divide(array_count);
    // compute Ht
    var ht = ee.Number(entropyaccumulation.get(index));
    
    // compute H_value
    var H_value_1 = pt.multiply(ee.Number(1).subtract(pt)).log();
    var H_value_2 = ht.divide(pt);
    var H_value_3 = H.subtract(ht).divide(ee.Number(1).subtract(pt));
    var H_value = H_value_1.add(H_value_2).add(H_value_3);
    return H_value;

  });
  // print(subentList);
  var _arr = ee.Array(subentList).multiply(1000).round();
  // print(_arr);
  var _arrmax = _arr.reduce(ee.Reducer.max(), [0]).get([0]);
  var maxindex = _arr.toList().indexOf(_arrmax);
  // var threshold = maxindex.multiply(minInternal).add(min); // the easy way
  var threshold = bucketMeans.toList().get(ee.Number(maxindex));  // the good way
  return threshold;

};


var props = function(class_t1, classnames, image_t1, image_t2, bandnames){
  var class_count = ee.List(classnames).length();
  var propListImageList = ee.List.sequence(1,class_count).map(function(class_num)
  {
    class_num = ee.Number(class_num);
    var band_count = ee.List(bandnames).length();
    var cell_count = class_t1.eq(ee.Image(class_num)).reduceRegion(ee.Reducer.sum()).get('class');
    var image_t1_clip = image_t1.updateMask(class_t1.eq(ee.Image(class_num)));
    var mean = image_t1_clip.reduceRegion(ee.Reducer.mean());
    var meanList = bandnames.map(function(n){
      var bandname = ee.String(n);
      return ee.Number(mean.get(bandname));
    });
    var variance = image_t1_clip.reduceRegion(ee.Reducer.variance());
    var mvariance = bandnames.map(function(a){
      var bandname_1 = ee.String(a);
      var bandindex_1 = bandnames.indexOf(a);
      var bandmean_1 = ee.Number(mean.get(bandname_1));
      var list = bandnames.map(function(b){
        var bandname_2 = ee.String(b);
        var bandindex_2 = bandnames.indexOf(b);
        var bandmean_2 = ee.Number(mean.get(bandname_2));
        var m1 = image_t1_clip.select([bandname_1], ['res']).subtract(ee.Image(bandmean_1));
        var m2 = image_t1_clip.select([bandname_2], ['res']).subtract(ee.Image(bandmean_2));
        var res = m1.multiply(m2).reduceRegion(ee.Reducer.mean()).get('res');
        return res;
      });
      list[bandindex_1] = ee.Number(variance.get(bandname_1));
      return list;
    });
    var s1 = image_t2.toArray().toArray(1).matrixTranspose().subtract(ee.Array([meanList]));
    var s2 = s1.matrixMultiply(ee.Array(mvariance).matrixInverse()).matrixMultiply(s1.matrixTranspose()).arrayGet([0, 0]).multiply(-0.5).exp();
    var coefficient = ee.Number(1).divide(ee.Number(2 * 3.1415926).pow(band_count).multiply(ee.Array(mvariance).matrixDeterminant().abs()).pow(0.5));
    var prop = s2.multiply(coefficient);
    return prop;
  });
  var propCollection = ee.ImageCollection(propListImageList);
  var propimage = propCollection.toArray().arrayProject([0]).arrayFlatten([classnames]);
  propimage = propimage.add(class_t1.multiply(0));

  var constantarray = ee.Array([ee.List.repeat(1.0,class_count)]);
  var sumimage = ee.Image(constantarray).matrixMultiply(propimage.toArray().toArray(1)).arrayGet([0,0]);
  propimage = propimage.divide(sumimage);
  
  return propimage;
};

var cvaps = function(class_t1, classnames, image_t1, image_t2, image_bands) {
  
  var props_t1 = props(class_t1, classnames, image_t1, image_t1, image_bands);
  var props_t2 = props(class_t1, classnames, image_t1, image_t2, image_bands);
    
  var sdcollectionList = ee.List(classnames).map(function(name){
    var class_name = ee.String(name);
    var prop_t1 = props_t1.select([class_name]);
    var prop_t2 = props_t2.select([class_name]);
    var differ = prop_t1.subtract(prop_t2);
    var square = differ.multiply(differ);
    return square.select([class_name],['array']);
  });
  
  var sdcollection = ee.ImageCollection(sdcollectionList);
  var sd = sdcollection.toArray().arrayProject([0]).arrayFlatten([classnames]);
  sd = sd.add(class_t1.multiply(0));
  
  var constantarray = ee.Array([ee.List.repeat(1.0,ee.List(classnames).length())]);
  sd = ee.Image(constantarray).matrixMultiply(sd.toArray().toArray(1)).arrayGet([0,0]);
  sd = sd.sqrt();
  
  // print(sd);
  // Map.addLayer(sd);
  // print(sd.getDownloadURL({'crs': 'EPSG:4326', 'name': 'sd'}));
  // Export.image.toAsset({"image":sd, "description": "temp_sd"});
  
  // var thresholdvalue = threshold(sd,"constant");
  var thresholdvalue = 1.0;
  print(thresholdvalue);
  var changedarea = sd.gt(ee.Image(ee.Number(thresholdvalue)));
  Map.addLayer(changedarea, {min: 0, max: 1, palette: ['00ff00', 'ff0000']},'changedarea');
  // print(changedarea);
  return changedarea;
}


var uncvaps = function(class_t1, classnames, image_t1, image_t2, image_bands){
  // classification 
  var sample_count = 2000;
  var samples = image_t1.addBands(class_t1).sample({
      numPixels: sample_count,
      seed: 0
  });
  var trained = ee.Classifier.cart().train(samples, 'class', image_bands);
  var classified_t1 = image_t1.select(image_bands).classify(trained);
  var classified_t2 = image_t2.select(image_bands).classify(trained);
  
  var changedarea = cvaps(class_t1, classnames, image_t1, image_t2, image_bands);
  
  var unchangedarea = ee.Image(ee.Number(1)).subtract(changedarea);
  
  var unchanged = class_t1.multiply(unchangedarea);
  var changed = classified_t2.select(['classification'], ['class']).multiply(changedarea);
  
  var class_t2 = class_t1.select('class').multiply(0.0);

  class_t2 = class_t2.add(unchanged.add(changed));

  class_t2 = class_t2.toInt32();
  // class_t2 = class_t2.expression("(class < 0) ? 0 : class", {'class': class_t2.select('class')});
  
  return class_t2;
};


var classPalette = ['1E90FF', // water
                    '708090', // urban
                    '2E8B57', // forest
                    '00FF7F'  // agriculture
                  ];  

var classnames = ['water', 'urban', 'forest', 'agriculture'];  

var image_bands = ['b1', 'b2', 'b3'];
var image_t1 = ee.Image('users/cas/cvaps/ref_t1').select(image_bands);
var image_t2 = ee.Image('users/cas/cvaps/ref_t2').select(image_bands);

var class_bands = ['b1'];
var class_t1 = ee.Image('users/cas/cvaps/class_t1').select(class_bands, ['class']);
Map.addLayer(class_t1, {min: 1, max: 4, palette: classPalette},'class_t1');

// var changedarea = uncvaps(class_t1, classnames, image_t1, image_t2, image_bands);
// Map.addLayer(changedarea, {min: 0, max: 1, palette: ['00ff00', 'ff0000']},'changedarea');

var class_t2 = uncvaps(class_t1, classnames, image_t1, image_t2, image_bands);
Map.addLayer(class_t2, {min: 1, max: 4, palette: classPalette},'class_t2');

// Export.image.toDrive(class_t2, "class_t2");
// print(class_t2.getDownloadURL({'crs': 'EPSG:4326'}));

Map.centerObject(class_t1);
