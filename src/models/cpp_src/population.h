#pragma once

#undef NDEBUG

#include <cstddef>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <vector>
#include <algorithm>

#include "csv_parser/csv.h"

namespace mocos_cpp
{

class Person
{
    inline static std::size_t count = 0;
 public:
    // Const parameters of person, which are not expected to change during simulation are accessible directly
    const std::size_t id;
    const std::size_t csv_index;
    const std::uint_fast16_t age;
    const std::uint_fast16_t gender;
    const std::size_t household_index;
    const std::string infection_status; // Uhhh, this will probably change...
    const std::size_t employment_status;
    const double social_competence;
    const std::uint_fast16_t public_transport_usage;
    const double public_transport_duration;
 private:
    // Parameters of person which WILL change during simulation are private and accessible only through getters/setters
    DetectionStatus detection_status;
 public:
    // Getters and setters for the above parameters
    DetectionStatus getDetectionStatus() const { return detection_status; };
    void setDetectionStatus(DetectionStatus ds) { detection_status = ds; };
 public:
    Person( std::size_t _csv_index,
            std::uint_fast16_t _age,
            std::uint_fast16_t _gender,
            std::size_t _household_index,
            const std::string& _infection_status,
            std::size_t _employment_status,
            double _social_competence,
            std::uint_fast16_t _public_transport_usage,
            double _public_transport_duration);

    void print(std::ostream& os = std::cout);
};

class InitialPopulation
{
    const std::vector<Person> population;
    std::vector<Person> get_pop_vector(const std::string& path);
    std::unordered_map<size_t, size_t> csv_id_map;
 public:
    InitialPopulation(const std::string& path);
    size_t size() const { return population.size(); };
    const Person& operator[](size_t idx) { return population[idx]; };
    const Person& by_csv_id(size_t idx) { if(csv_id_map.count(idx) > 0) return population[csv_id_map[idx]]; else throw std::logic_error("Referenced nonexistent person CSV ID"); };
};





// =========================== IMPLEMENTATION ========================
//

Person::Person( std::size_t _csv_index,
                std::uint_fast16_t _age,
                std::uint_fast16_t _gender,
                std::size_t _household_index,
                const std::string& _infection_status,
                std::size_t _employment_status,
                double _social_competence,
                std::uint_fast16_t _public_transport_usage,
                double _public_transport_duration
) :
id(count++),
csv_index(_csv_index),
age(_age),
gender(_gender),
household_index(_household_index),
infection_status(_infection_status),
employment_status(_employment_status),
social_competence(_social_competence),
public_transport_usage(_public_transport_usage),
public_transport_duration(_public_transport_duration),
detection_status(NotDetected)
{
    assert(id < 10000000000);
    assert(age < 150);
    assert(gender < 2);
}


#define PRINT_FIELD(F) os << #F << ": " << F << std::endl;
void Person::print(std::ostream& os)
{
    PRINT_FIELD(id);
    PRINT_FIELD(csv_index);
    PRINT_FIELD(age);
    PRINT_FIELD(gender);
    PRINT_FIELD(household_index);
    PRINT_FIELD(infection_status);
    PRINT_FIELD(employment_status);
    PRINT_FIELD(social_competence);
    PRINT_FIELD(public_transport_usage);
    PRINT_FIELD(public_transport_duration);
}

InitialPopulation::InitialPopulation(const std::string& path) :
population(get_pop_vector(path))
{
    for(const Person& p : population)
        csv_id_map[p.csv_index] = p.id;

};

std::vector<Person> InitialPopulation::get_pop_vector(const std::string& path)
{
    //std::cout << path << std::endl;
    io::CSVReader<9> csv_reader(path);

    std::vector<Person> ret;

    std::size_t idx;
    std::uint_fast16_t age;
    std::uint_fast16_t gender;
    std::size_t household_index;
    std::string infection_status;
    std::size_t employment_status;
    double social_competence;
    std::uint_fast16_t public_transport_usage;
    double public_transport_duration;

    csv_reader.read_header(io::ignore_extra_column, "idx", "age", "gender", "household_index", "infection_status", "employment_status", "social_competence", "public_transport_usage", "public_transport_duration");

    while(csv_reader.read_row(idx, age, gender, household_index, infection_status, employment_status, social_competence, public_transport_usage, public_transport_duration))
        ret.emplace_back(Person(idx, age, gender, household_index, infection_status, employment_status, social_competence, public_transport_usage, public_transport_duration));

    return ret;
}

} // namespace mocos_cpp
