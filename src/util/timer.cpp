#include "timer.hpp"
#include <iostream>

Timer::Timer(std::string slug) : slug(slug) {}

void Timer::start() { c_start = std::clock(); }

void Timer::stop() { c_end = std::clock(); }

double Timer::get_ms() { return 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC; }

void Timer::print_info() {
    std::cerr.precision(4);
    std::cerr << std::fixed << slug << ": " << get_ms() << " ms." << std::endl;
}
