#include "ETL/ETL.h"

#include <iostream>
#include <string>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>

int main(int argc, char *argv[])
{
    ETL etl(argv[1], argv[2], argv[3]);
    std::vector<std::vector<std::string>> dataset = etl.readCSV();

    int rows = dataset.size();
    int cols = dataset[0].size();

    Eigen::MatrixXd dataMat = etl.CSVtoEigen(dataset, rows, cols);
    // Eigen::MatrixXd norm = etl.Normalize(dataMat);
    Eigen::MatrixXd norm2 = etl.Norm(dataMat);

    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> split_data = etl.TrainTestSplit(norm2, 0.8);
    std::tie(X_train, y_train, X_test, y_test) = split_data;

    std::cout << "X_train rows: " << X_train.rows() << std::endl;
    std::cout << "X_train cols: " << X_train.cols() << std::endl;
    std::cout << "y_train rows: " << y_train.rows() << std::endl;
    std::cout << "y_train cols: " << y_train.cols() << std::endl;
    std::cout << "X_test rows: " << X_test.rows() << std::endl;
    std::cout << "X_test cols: " << X_test.cols() << std::endl;
    std::cout << "y_test rows: " << y_test.rows() << std::endl;
    std::cout << "y_test cols: " << y_test.cols() << std::endl;

    return EXIT_SUCCESS;
}