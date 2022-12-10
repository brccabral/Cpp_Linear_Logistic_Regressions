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
};