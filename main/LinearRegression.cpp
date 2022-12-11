#include "../ETL/ETL.h"
#include "../LinearRegression/LinearRegression.h"

#include <iostream>
#include <string>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>

int main(int argc, char *argv[])
{
    ETL etl(argv[1], argv[2], argv[3]);
    LinearRegression lr;

    std::vector<std::vector<std::string>> dataset = etl.readCSV();

    int rows = dataset.size();
    int cols = dataset[0].size();

    std::cout << "Rows: " << rows << std::endl;
    std::cout << "Cols: " << cols << std::endl;

    Eigen::MatrixXd dataMat = etl.CSVtoEigen(dataset, rows, cols);

    std::cout << "dataMat (" << dataMat.rows() << "," << dataMat.cols() << ")" << std::endl;
    Eigen::MatrixXd norm = etl.Normalize(dataMat, true);

    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> split_data = etl.TrainTestSplit(norm, 0.8);
    std::tie(X_train, y_train, X_test, y_test) = split_data;

    std::cout << "X_train (" << X_train.rows() << "," << X_train.cols() << ")" << std::endl;

    Eigen::VectorXd vec_train = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vec_test = Eigen::VectorXd::Ones(X_test.rows());

    X_train.conservativeResize(X_train.rows(), X_train.cols() + 1);
    X_train.col(X_train.cols() - 1) = vec_train;

    X_test.conservativeResize(X_test.rows(), X_test.cols() + 1);
    X_test.col(X_test.cols() - 1) = vec_test;

    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());

    float alpha = 0.01;
    int iters = 1000;

    Eigen::VectorXd thetaOut;
    std::vector<float> cost;

    std::tuple<Eigen::VectorXd, std::vector<float>> gd = lr.GradientDescent(X_train, y_train, theta, alpha, iters);
    std::tie(thetaOut, cost) = gd;

    std::cout << "thetaOut: " << thetaOut << std::endl;

    etl.VectorToFile(cost, "dataset/cost.txt");
    etl.EigenToFile(thetaOut, "dataset/thetaOut.txt");

    auto mu_data = etl.Mean(dataMat);
    auto mu_z = mu_data(0, 11);

    auto scaled_data = dataMat.rowwise() - dataMat.colwise().mean();

    auto sigma_data = etl.Std(scaled_data);
    auto sigma_z = sigma_data(0, 11);

    Eigen::MatrixXd y_train_hat = (X_train * thetaOut * sigma_z).array() + mu_z;
    Eigen::MatrixXd y = dataMat.col(11).topRows(y_train_hat.rows());

    float R_Squared = lr.RSquared(y, y_train_hat);
    std::cout << "R-Squared: " << R_Squared << std::endl;

    etl.EigenToFile(y_train_hat, "dataset/y_train_hat.txt");

    return EXIT_SUCCESS;
}