#include "ETL.h"

#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

std::vector<std::vector<std::string>> ETL::readCSV()
{
    std::ifstream file(dataset);
    if (!file.is_open())
    {
        std::cout << "Can't open file " << dataset << std::endl;
    }
    std::vector<std::vector<std::string>> dataString;

    std::string line = "";

    while (getline(file, line))
    {
        std::vector<std::string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimiter));
        dataString.push_back(vec);
    }

    file.close();

    return dataString;
}

Eigen::MatrixXd ETL::CSVtoEigen(std::vector<std::vector<std::string>> dataset, int rows, int cols)
{
    if (header == true)
    {
        rows = rows - 1;
    }

    Eigen::MatrixXd mat(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            mat(j, i) = atof(dataset[i][j].c_str());
        }
    }

    return mat.transpose();
}

auto ETL::Mean(Eigen::MatrixXd data) -> decltype(data.colwise().mean())
{
    return data.colwise().mean();
}

auto ETL::Std(Eigen::MatrixXd data) -> decltype((data.array().square().colwise().sum() / (data.rows() - 1)).sqrt())
{
    return ((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt();
}

// normalize data
Eigen::MatrixXd ETL::Normalize(Eigen::MatrixXd data, bool normalizeTarget)
{
    // when target is category we don't need to normalize it

    Eigen::MatrixXd dataNorm;
    if (normalizeTarget)
        dataNorm = data;
    else
        dataNorm = data.leftCols(data.cols() - 1);

    // although data.colwise().mean() is used in the next line,
    // for some reason the mean variable has to remain in the code
    auto mean = Mean(dataNorm);
    // cannot use mean variable in the next line because of floating point overflow
    // eigen library deals with overflow problems
    Eigen::MatrixXd scaled_data = dataNorm.rowwise() - dataNorm.colwise().mean();
    // to use Eigen3.4 need to not use std variable
    // auto std = Std(scaled_data);
    // Eigen::MatrixXd norm = scaled_data.array().rowwise() / std;
    Eigen::MatrixXd norm = scaled_data.array().rowwise() / (((scaled_data.array().square().colwise().sum()) / (scaled_data.rows() - 1)).sqrt());

    if (!normalizeTarget)
    {
        norm.conservativeResize(norm.rows(), norm.cols() + 1);
        norm.col(norm.cols() - 1) = data.rightCols(1);
    }

    return norm;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ETL::TrainTestSplit(Eigen::MatrixXd data, float train_size)
{
    int rows = data.rows();
    int train_rows = round(train_size * rows);
    int test_rows = rows - train_rows;

    Eigen::MatrixXd train = data.topRows(train_rows);
    Eigen::MatrixXd X_train = train.leftCols(data.cols() - 1);
    Eigen::MatrixXd y_train = train.rightCols(1);

    Eigen::MatrixXd test = data.bottomRows(test_rows);
    Eigen::MatrixXd X_test = test.leftCols(data.cols() - 1);
    Eigen::MatrixXd y_test = test.rightCols(1);

    return std::make_tuple(X_train, y_train, X_test, y_test);
}

void ETL::VectorToFile(std::vector<float> vector, std::string filename)
{
    std::ofstream output_file(filename);
    std::ostream_iterator<float> output_iterator(output_file, "\n");
    std::copy(vector.begin(), vector.end(), output_iterator);
}

void ETL::EigenToFile(Eigen::MatrixXd data, std::string filename)
{
    std::ofstream output_file(filename);
    if (output_file.is_open())
    {
        output_file << data << "\n";
    }
}