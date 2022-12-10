#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Dense>

class ETL
{
    std::string dataset;
    std::string delimiter;
    bool header;

public:
    ETL(std::string data, std::string separator, bool head) : dataset(data), delimiter(separator), header(head)
    {
    }
    std::vector<std::vector<std::string>> readCSV();
    Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> dataset, int rows, int cols);

    auto Mean(Eigen::MatrixXd data) -> decltype(data.colwise().mean());
    auto Std(Eigen::MatrixXd data) -> decltype((data.array().square().colwise().sum() / (data.rows() - 1)).sqrt());
    Eigen::MatrixXd Normalize(Eigen::MatrixXd data);
    Eigen::MatrixXd Norm(Eigen::MatrixXd data);

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd data, float train_size);

    void VectorToFile(std::vector<float> vector, std::string filename);
    void EigenToFile(Eigen::MatrixXd data, std::string filename);
};