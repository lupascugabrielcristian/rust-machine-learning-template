use csv::Reader;
use std::fs::File;
use ndarray::{ Array, Array1, Array2 };
use linfa::Dataset;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};

fn get_dataset() -> Dataset<f32, i32, ndarray::Dim<[usize;1]>> {
    let mut reader = Reader::from_path("./src/heart.csv").unwrap();

    let headers = get_headers(&mut reader);
    let data = get_data(&mut reader);
    let target_index = headers.len() - 1;

    let features = headers[0..target_index].to_vec();
    let records = get_records(&data, target_index);
    let targets = get_targets(&data, target_index);

    return Dataset::new(records, targets).with_feature_names(features);
}

fn get_headers( reader: &mut Reader<File>) -> Vec<String> {
    return reader
        .headers().unwrap().iter()
        .map(|r| r.to_owned())
        .collect();
}

fn get_data(reader: &mut Reader<File>) -> Vec<Vec<f32>> {
    return reader
        .records()
        .map( |r|
              r.unwrap().iter()
              .map(|field| field.parse::<f32>().unwrap())
              .collect::<Vec<f32>>()
              )
        .collect::<Vec<Vec<f32>>>();

}

fn get_records(data: &Vec<Vec<f32>>, target_index: usize) -> Array2<f32> {
    let mut records: Vec<f32> = vec![];
    for record in data.iter() {
        records.extend_from_slice( &record[0..target_index] );
    }

    return Array::from( records ).into_shape((1025, 13)).unwrap();
}

fn get_targets(data: &Vec<Vec<f32>>, target_index: usize ) -> Array1<i32> {
    let targets = data
        .iter()
        .map(|record| record[target_index] as i32)
        .collect::<Vec<i32>>();
    return Array::from( targets );
}

fn plot_data(dataset: &Dataset<f32, i32, ndarray::Dim<[usize;1]>>) {
    let records = dataset.records().clone().into_raw_vec();
    let _features = dataset.feature_names();
    //Toate valorile intr-un singur sir
    println!("{:?} records", records.len());

    // Toate valorile in bucati de cate 13 - atatea sunt pe un singur rand de date
    // chunks face o lista in care fiecare element este o lista cu 13 elemente
    let chunks: Vec<&[f32]> = records.chunks(13).collect();
    println!("{:?} chunks", chunks.len());

    // Targets nu face parte din records
    let targets = dataset.targets().clone().into_raw_vec();
    println!("{:?} targets", targets.len());
    let mut positive = vec![];
    let mut negative = vec![];
    for i in 0..chunks.len() {
        let current_row = chunks.get(i).expect("current row");
        // Daca target pentru fiecare rand este 1, adaug la positive, valoarea corespunzatoare
        // features trestbps, index 3
        if let Some(&1) = targets.get(i) {
            positive.push(( current_row[3], 1 ));
        } else {
            negative.push(( current_row[3], 0 ))
        }
    }

    println!("positive {:?}", positive.len());
    println!("negative {:?}", negative.len());
    
    // Aici imi da eroare pentru ca mie mi-ar trebui f64 in loc de f32 pentru valorile din
    // current_row
    let plot_positive = Plot::new(positive)
        .point_style(
            PointStyle::new()
                .size(2.0)
                .marker(PointMarker::Square)
                .colour("#00ff00"),
            )
        .legend("Trestbps".to_string());

    let plot_negative = Plot::new(negative)
        .point_style(
            PointStyle::new()
                .size(2.0)
                .marker(PointMarker::Square)
                .colour("#ff0000"),
            );
}

fn main() {
    let dataset = get_dataset();
    println!("{:?}", dataset);
    plot_data(&dataset);

    let (train, test) = dataset.split_with_ratio(0.9);
    println!("{:?} Train records", train.records.len());
    println!("{:?} Test records", test.targets.len());

    //let model = DecisionTree::params().fit(&train).unwrap();
    //let predictions = model.predict(&test);

    //println!("{:?}", predictions);
    //println!("{:?}", test.targets);

}

