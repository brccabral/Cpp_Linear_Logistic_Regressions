#include "../ETL/ETL.h"

#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>

int main(int argc, char *argv[])
{
    ETL etl(argv[1], argv[2], argv[3]);

    std::vector<std::vector<std::string>> dataset = etl.readCSV();

    int rows = dataset.size();
    int cols = dataset[0].size();

    Eigen::MatrixXd dataMat = etl.CSVtoEigen(dataset, rows, cols);

    Eigen::MatrixXd norm = etl.Normalize(dataMat, false);

    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> split_data = etl.TrainTestSplit(norm, 0.8);
    std::tie(X_train, y_train, X_test, y_test) = split_data;

    std::cout << "X_train shape(" << X_train.rows() << "," << X_train.cols() << ")" << std::endl;
    std::cout << "y_train shape(" << y_train.rows() << "," << y_train.cols() << ")" << std::endl;
    std::cout << "X_test shape(" << X_test.rows() << "," << X_test.cols() << ")" << std::endl;
    std::cout << "y_test shape(" << y_test.rows() << "," << y_test.cols() << ")" << std::endl;

    return EXIT_SUCCESS;
}