#pragma once

#include <eigen3/Eigen/Dense>
#include <list>

class LogisticRegression
{
public:
    LogisticRegression() {}
    Eigen::MatrixXd Sigmoid(Eigen::MatrixXd X);
};