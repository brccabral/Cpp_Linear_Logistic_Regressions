#pragma once

#include <eigen3/Eigen/Dense>
#include <list>

class LogisticRegression
{
public:
    LogisticRegression() {}
    Eigen::MatrixXd Sigmoid(Eigen::MatrixXd X);
    std::tuple<Eigen::MatrixXd, double, double> Propagate(Eigen::MatrixXd W, double b, Eigen::MatrixXd X, Eigen::MatrixXd y, double lambda);
};