#include "timer.hpp"
#include <iostream>

Timer::Timer(std::string slug) : slug(slug), c_start(std::clock()) {}

void Timer::start() { c_start = std::clock(); }

void Timer::stop() { c_end = std::clock(); }

double Timer::get_ms() { return 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC; }

void Timer::print_info(std::string desc) {
    std::cout.precision(3);
    std::cout << std::fixed << slug << ": " << desc << ": " << get_ms() << " ms." << std::endl;
}

void Timer::print_and_restart(std::string desc) {
    stop();
    print_info(desc);
    start();
}
