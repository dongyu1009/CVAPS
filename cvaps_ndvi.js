var boundtable = beijing;
function sgfilter(start_date,end_date, imagecollection){
  
  var origin = ee.ImageCollection(imagecollection).
              filterDate(start_date, end_date).sort('system:time_start');
              
  var prepared = origin.map(function(img) {
    var dstamp = ee.Date(img.get('system:time_start'))
    var ddiff = dstamp.difference(ee.Date(start_date), 'hour')
    img = img.select(['NDVI']).set('date', dstamp)
    return img.addBands(ee.Image(1).toFloat().rename('constant')).
      addBands(ee.Image(ddiff).toFloat().rename('t')).
      addBands(ee.Image(ddiff).pow(ee.Image(2)).toFloat().rename('t2')).
      addBands(ee.Image(ddiff).pow(ee.Image(3)).toFloat().rename('t3'))
  })

  // Step 2: Set up Savitzky-Golay smoothing
  var window_size = 9
  var half_window = (window_size - 1)/2
  
  // Define the axes of variation in the collection array.
  var imageAxis = 0;
  var bandAxis = 1;
  
  // Set polynomial order
  var order = 3
  var coeffFlattener = [['constant', 'x', 'x2', 'x3']]
  var indepSelectors = ['constant', 't', 't2', 't3']
  
  var array = prepared.toArray();
  
  
  // Solve 
  function getLocalFit(i) {
    // Get a slice corresponding to the window_size of the SG smoother
    var subarray = array.arraySlice(imageAxis, ee.Number(i).int(), ee.Number(i).add(window_size).int())
    var predictors = subarray.arraySlice(bandAxis, 1, 1 + order + 1)
    var response = subarray.arraySlice(bandAxis, 0, 1); // ndvi
    var coeff = predictors.matrixSolve(response)
  
    coeff = coeff.arrayProject([0]).arrayFlatten(coeffFlattener)
    return coeff  
  }
  
  
  // For the remainder, use s1res as a list of images
  prepared = prepared.toList(prepared.size())
  var runLength = ee.List.sequence(0, prepared.size().subtract(window_size))
  
  // Run the SG solver over the series, and return the smoothed image version
  var sg_series = runLength.map(function(i) {
    var ref = ee.Image(prepared.get(ee.Number(i).add(half_window)))
    return getLocalFit(i).multiply(ref.select(indepSelectors)).reduce(ee.Reducer.sum()).copyProperties(ref)
  })
  
  sg_series = ee.ImageCollection(sg_series);
  
  return sg_series;
}

function canberradistance(){
  var bound = ee.Feature(boundtable.first()).geometry();
  // Map.centerObject(bound);

  var bandnames = ee.List.sequence(1,46).map(function(n) {
    var number = ee.Number(n);
    return ee.String('b').cat(number.int());
  });
  
  var l8_start_date = '2015-01-01';
  var l8_end_date = '2015-12-31';
  // var landsat8ndvi = sgfilter(l8_start_date, l8_end_date, "LANDSAT/LC8_L1T_8DAY_NDVI");
  var landsat8ndvi = ee.ImageCollection("LANDSAT/LC8_L1T_8DAY_NDVI").filterDate(l8_start_date, l8_end_date);
  landsat8ndvi = landsat8ndvi.map(function(n){
    var img = ee.Image(n);
    img = img.clip(bound);
    img = img.expression('img ? img : -100', {'img': img.select('NDVI')});
    return img;
  });
  landsat8ndvi = ee.ImageCollection(landsat8ndvi);

  var i8ndvi = landsat8ndvi.toArray().arrayProject([0]).arrayFlatten([bandnames]);

  var l7_start_date = '2009-1-1';
  var l7_end_date = '2009-12-31';
  // var landsat7ndvi = sgfilter(l7_start_date, l7_end_date, "LANDSAT/LT5_L1T_8DAY_NDVI");
  var landsat7ndvi = ee.ImageCollection("LANDSAT/LT5_L1T_8DAY_NDVI").filterDate(l7_start_date, l7_end_date);
  landsat7ndvi = landsat7ndvi.map(function(n){
    var img = ee.Image(n);
    img = img.clip(bound);
    img = img.expression('img ? img : -100', {'img': img.select('NDVI')});
    return img;
  });
  landsat7ndvi = ee.ImageCollection(landsat7ndvi);
  var i7ndvi = landsat7ndvi.toArray().arrayProject([0]).arrayFlatten([bandnames]);

  var res = ee.List.sequence(1,12).map(function(n) {
    var bandname = ee.String('b').cat(ee.Number(n).int());
    var ndvi1 = i7ndvi.select([bandname],['ndvi']);
    var ndvi2 = i8ndvi.select([bandname],['ndvi']);
    var normalization = ndvi1.abs().add(ndvi2.abs());
    var division = ndvi1.subtract(ndvi2).abs();
    var res = division.divide(normalization)
    res = res.expression('ndvi1 < 0.05 || ndvi2 < 0.05 ? 0 : res', {
            'ndvi1': ndvi1.select('ndvi') , 
            'ndvi2': ndvi2.select('ndvi') , 
            'res': res.select('ndvi') 
    });
    return res;
  });
  res = ee.ImageCollection(res);
  res = res.mean();
  var imagebound = ee.Image("users/cas/cvaps/westofchina_bound_outer");
  res = imagebound.add(res.select(['ndvi'], ['b1']));
  res = res.clip(bound);
  // print(res);
  // Export.image.toAsset({ 'image' : res, 'description': 'ndvi_cd', 'region' : bound,'maxPixels' : 30000000000});
  // Map.addLayer(res, { min: 0, max: 1,}, 'cd');
  return res;
}

var threshold = function(image,bandname)
{
  var internal = 0.1;
  image = image.select([bandname]);
  var array_count = image.multiply(0.0).add(1.0).reduceRegion(ee.Reducer.sum()).get(bandname);
  array_count = ee.Number(array_count);
  // print(array_count);
  var min = image.reduceRegion(ee.Reducer.min()).get(bandname);
  min = ee.Number(min);
  var max = image.reduceRegion(ee.Reducer.max()).get(bandname);
  max = ee.Number(max);

  var accumList = ee.List.sequence(min, max.add(internal), internal).map(function(n) {
    var number = ee.Number(n);
    var srr = image.lt(number);
    var srrsum = srr.reduceRegion(ee.Reducer.sum()).get(bandname);
    return ee.Number(srrsum);
  });
  var s1 = accumList.slice(0,accumList.length().subtract(1))
  var s2 = accumList.slice(1,accumList.length())
  
  var countArray = ee.Array(s2).subtract(ee.Array(s1));

  var internalcount = countArray.length().get([0]);
  var propArray = countArray.divide(ee.List.repeat(array_count,internalcount));
  var entropy = propArray.add(propArray.eq(0)).log().multiply(propArray);
  entropy = ee.Array(ee.List.repeat(0, internalcount)).subtract(entropy);
  var H = entropy.reduce(ee.Reducer.sum(), [0]).get([0]);
  var subentList = ee.List.sequence(min, max, internal).map(function(n) {
    var number = ee.Number(n);
    var index = number.subtract(min).divide(internal).round();
    
    // compute Pt
    var srr = image.lt(number);
    var srrsum = srr.reduceRegion(ee.Reducer.sum()).get(bandname);
    var pt = ee.Number(srrsum).divide(array_count);
    // compute Ht
    var sliceentropy = entropy.slice({start: 0, end: index.add(1)});
    var ht = sliceentropy.reduce(ee.Reducer.sum(), [0]).get([0]);
    
    // compute H_value
    var H_value_1 = pt.multiply(ee.Number(1).subtract(pt)).log();
    var H_value_2 = ht.divide(pt);
    var H_value_3 = H.subtract(ht).divide(ee.Number(1).subtract(pt));
    var H_value = H_value_1.add(H_value_2).add(H_value_3);
    return H_value;
  });
  var _arr = ee.Array(subentList).multiply(1000).round();
  var _arrmax = _arr.reduce(ee.Reducer.max(), [0]).get([0]);
  var maxindex = _arr.toList().indexOf(_arrmax);
  var threshold = maxindex.multiply(internal).add(min);
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

var uncvaps = function(class_t1, class_names, class_nums, image_t1, image_t2, image_bands){
  // algorithm
  var sample_count = 5000;
  var samples = image_t1.addBands(class_t1).sample({
      numPixels: sample_count,
      seed: 0
  });
  
  var trained = ee.Classifier.cart().train(samples, 'class', image_bands);
  var classified_t1 = image_t1.select(image_bands).classify(trained);
  var classified_t2 = image_t2.select(image_bands).classify(trained);
  
  print(classified_t2);
  Map.addLayer(classified_t2);

  var props_t1 = props(class_t1, class_names, image_t1, image_t1, image_bands);
  var props_t2 = props(class_t1, class_names, image_t1, image_t2, image_bands);
  
  // Export.image.toAsset({ 'image' : props_t1, 'description': 'props_t1', 'maxPixels' : 30000000000});
  
  var sdcollectionList = ee.List(class_names).map(function(name){
    var class_name = ee.String(name);
    var prop_t1 = props_t1.select([class_name]);
    var prop_t2 = props_t2.select([class_name]);
    var differ = prop_t1.subtract(prop_t2);
    var square = differ.multiply(differ);
    return square.select([class_name],['array']);
  });
  
  var sdcollection = ee.ImageCollection(sdcollectionList);
  var sd = sdcollection.toArray().arrayProject([0]).arrayFlatten([class_names]);
  sd = sd.add(class_t1.multiply(0));
  
  var constantarray = ee.Array([ee.List.repeat(1.0,ee.List(class_names).length())]);
  sd = ee.Image(constantarray).matrixMultiply(sd.toArray().toArray(1)).arrayGet([0,0]);
  sd = sd.sqrt();
  // Map.addLayer(sd);
  // var bound1 = ee.Geometry.Rectangle([106.951224, 34.299043, 107.497969, 34.445399]);
  // var bound1 = ee.Geometry.Rectangle([80.001224, 37.5, 80.669, 38.1]);
  var bound = ee.Feature(boundtable.first()).geometry();
  Export.image.toAsset({ 'image' : sd, 'description': 'sd', 'region' : geometry, 'maxPixels' : 30000000000});
  // print(sd.getDownloadURL({'crs': 'EPSG:4326'}));
  var thresholdvalue = threshold(ee.Image('users/cas/cvaps/westofchina_bound_outer').select(['b1'], ['constant']).add(sd),'constant');
  thresholdvalue = ee.Number(0.6);
  
  var unchange1 = sd.lt(class_t1.multiply(.0).add(ee.Image.constant(thresholdvalue)));
  var change1 = sd.gt(class_t1.multiply(.0).add(ee.Image.constant(thresholdvalue)));
  
  var cdndvi = canberradistance();
  var thresholdvalue2 = ee.Number(0.1);
  var unchange2 = cdndvi.lt(class_t1.multiply(.0).add(ee.Image.constant(thresholdvalue2)));
  var change2 = cdndvi.gt(class_t1.multiply(.0).add(ee.Image.constant(thresholdvalue2)));
  Export.image.toAsset({ 'image' : cdndvi, 'description': 'cdndvi', 'region' : geometry, 'maxPixels' : 30000000000});
  var change = change2.multiply(change1);
  var unchange = class_t1.multiply(0.0).add(1.0).subtract(change);
  
  var unchanged = class_t1.multiply(unchange);
  var changed = classified_t2.select(['classification'], ['class']).multiply(change);
  
  var class_t2 = class_t1.select(class_bands[0]).multiply(0.0);
  class_t2 = unchanged.add(changed);
  class_t2 = class_t2.toInt32();
  class_t2 = class_t2.expression("(class < 0) ? 0 : class", {'class': class_t2.select('class')});
  
  return class_t2;
    
};

// ***********************************************
// finished by Dong Yu from Igsnnr,CAS. 2017.12.10
// ***********************************************
// read data

var image_bands = ['b1', 'b2', 'b3'];
var image_t1 = ee.Image('users/cas/cvaps/ref_t1').select(image_bands);
var image_t2 = ee.Image('users/cas/cvaps/ref_t2').select(image_bands);

var class_bands = ['b1'];
var class_t1 = ee.Image('users/cas/cvaps/class_t1').select(class_bands, ['class']);
var class_nums = [1,2,3,4];
var classPalette = ['00FF7F',  // agriculture
                    '2E8B57', // forest
                    '9dffce', // grass
                    '1E90FF', // water
                    '708090', // urban
                    'FFFFFF' // bare
                  ];  
var class_names = ['agriculture', 'forest', 'grass', 'water', 'urban', 'bare'];  
var image_bands = ['B1', 'B2', 'B3'];
var class_bands = ['landcover'];
var class_nums = [1,2,3,4,5,6];
//var bound = ee.Geometry.Rectangle([106.951224, 34.299043, 107.497969, 34.445399]);
var bound = ee.Feature(boundtable.first()).geometry();

var image_t1 = ee.Image("LANDSAT/LE7_TOA_1YEAR/2009").select(image_bands);
// print(image_t1);
// Map.addLayer(image_t1);
var image_t2 = ee.Image("LANDSAT/LE7_TOA_1YEAR/2014").select(image_bands);
// print(image_t2);
var class_t1 = ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3").select(class_bands,  ['class']);
var targetprojection = class_t1.projection();
// var bound1 = ee.Geometry.Rectangle([106.951224, 34.299043, 107.497969, 34.445399]);
// Export.image.toAsset({ 'image' : class_t1, 'description': 'class_t1', 'region' : bound1, 'maxPixels' : 30000000000});
  
// print(class_t1);
// var oldclass = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22];
var oldclass = [11,14,20,30,40,50,60,70,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230];
var newclass = [1,1,1,1,2,2,2,2,2,2, 3, 3, 3, 3, 6, 4, 4, 4, 5, 6, 4, 4, 6];
Map.addLayer(bound, {}, 'china');
var image_t1 = image_t1.clip(bound);
var image_t2 = image_t2.clip(bound);

image_t1 = image_t1.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
    .reproject({crs: targetprojection});
image_t2 = image_t2.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
    .reproject({crs: targetprojection});
    
// var image_t1 = ee.Image('users/cas/cvaps/westofchina_bound_outer').multiply(.0).add(image_t1);
// var image_t2 = ee.Image('users/cas/cvaps/westofchina_bound_outer').multiply(.0).add(image_t2);

var class_t1 = class_t1.clip(bound);
class_t1 = class_t1.remap(oldclass,newclass).select(['remapped'],['class']);
// print('class_t1 : ',class_t1.getDownloadURL({'crs': 'EPSG:4326'}));
// print(class_t1.getDownloadURL({'crs': 'EPSG:4326'}));

var class_t2 = uncvaps(class_t1, class_names, class_nums, image_t1, image_t2, image_bands);

// show data
Map.centerObject(class_t1, 10);
Map.addLayer(class_t1, {min: 1, max: 6, palette: classPalette},'class_t1');
Map.addLayer(class_t2, {min: 1, max: 6, palette: classPalette},'class_t2');


// print(class_t2);
// print(class_t2.getDownloadURL({'crs': 'EPSG:4326'}));
Export.image.toAsset({ 'image' : class_t2, 'description': 'class_t2', 'region' : bound, 'maxPixels' : 30000000000});
// ndvi
/*
var ndvi_image = ee.Image("LANDSAT/LE7_TOA_1YEAR/2009");
ndvi_image = ndvi_image.clip(bound);
ndvi_image = ndvi_image.expression('(nir - red) / (nir + red)', {
        'red': ndvi_image.select('B3') , 
        'nir': ndvi_image.select('B4')
});

Export.image.toAsset({ 'image' : ndvi_image, 'description': 'ndvi_image', 'region' : bound, 'maxPixels' : 30000000000});
// print(ndvi_image);
// Map.addLayer(ndvi_image, {min: -1, max: 1, palette: ['FF0000', '00FF00']});
*/


