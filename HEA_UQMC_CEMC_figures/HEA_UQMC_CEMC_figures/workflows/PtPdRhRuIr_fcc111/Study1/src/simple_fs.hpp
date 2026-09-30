#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace simple_fs {

class path {
    std::string value_;
public:
    path() = default;
    path(const char *s) : value_(s ? s : "") {}
    path(const std::string &s) : value_(s) {}

    bool empty() const { return value_.empty(); }
    std::string string() const { return value_; }
    const char *c_str() const { return value_.c_str(); }
    operator std::string() const { return value_; }

    path parent_path() const {
        if (value_.empty()) return path();
        size_t end = value_.find_last_not_of('/');
        if (end == std::string::npos) return path("/");
        size_t pos = value_.find_last_of('/', end);
        if (pos == std::string::npos) return path();
        if (pos == 0) return path("/");
        return path(value_.substr(0, pos));
    }

    std::string extension() const {
        size_t slash = value_.find_last_of('/');
        size_t dot = value_.find_last_of('.');
        if (dot == std::string::npos) return "";
        if (slash != std::string::npos && dot < slash) return "";
        return value_.substr(dot);
    }

    path &operator+=(const std::string &s) {
        value_ += s;
        return *this;
    }

    friend path operator/(const path &a, const path &b) {
        if (a.value_.empty()) return b;
        if (b.value_.empty()) return a;
        if (b.value_[0] == '/') return b;
        if (a.value_.back() == '/') return path(a.value_ + b.value_);
        return path(a.value_ + "/" + b.value_);
    }

    friend path operator/(const path &a, const char *b) { return a / path(b); }
    friend bool operator<(const path &a, const path &b) { return a.value_ < b.value_; }
    friend bool operator==(const path &a, const path &b) { return a.value_ == b.value_; }
};

inline std::ostream &operator<<(std::ostream &os, const path &p) {
    os << p.string();
    return os;
}

inline bool exists(const path &p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

inline bool is_directory(const path &p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline bool is_regular_file_path(const path &p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

inline void create_directories(const path &p) {
    if (p.empty() || exists(p)) return;
    path parent = p.parent_path();
    if (!parent.empty() && !exists(parent)) create_directories(parent);
    if (::mkdir(p.c_str(), 0775) != 0 && errno != EEXIST) {
        throw std::runtime_error("Could not create directory: " + p.string() + " (" + std::strerror(errno) + ")");
    }
}

inline void remove(const path &p) {
    if (!exists(p)) return;
    if (::remove(p.c_str()) != 0 && errno != ENOENT) {
        throw std::runtime_error("Could not remove: " + p.string() + " (" + std::strerror(errno) + ")");
    }
}

inline void rename(const path &from, const path &to) {
    if (::rename(from.c_str(), to.c_str()) != 0) {
        throw std::runtime_error("Could not rename " + from.string() + " to " + to.string() + " (" + std::strerror(errno) + ")");
    }
}

inline unsigned long long file_size(const path &p) {
    struct stat st;
    if (::stat(p.c_str(), &st) != 0) return 0;
    return static_cast<unsigned long long>(st.st_size);
}

inline void remove_all(const path &p) {
    if (!exists(p)) return;
    if (!is_directory(p)) {
        remove(p);
        return;
    }
    DIR *dir = ::opendir(p.c_str());
    if (!dir) return;
    while (dirent *ent = ::readdir(dir)) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        remove_all(p / name);
    }
    ::closedir(dir);
    if (::rmdir(p.c_str()) != 0 && errno != ENOENT) {
        throw std::runtime_error("Could not remove directory: " + p.string() + " (" + std::strerror(errno) + ")");
    }
}

enum class copy_options { overwrite_existing };

inline void copy_file(const path &src, const path &dst, copy_options) {
    std::ifstream in(src.string(), std::ios::binary);
    if (!in) throw std::runtime_error("Could not open copy source: " + src.string());
    path parent = dst.parent_path();
    if (!parent.empty()) create_directories(parent);
    std::ofstream out(dst.string(), std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Could not open copy destination: " + dst.string());
    out << in.rdbuf();
}

class directory_entry {
    simple_fs::path p_;
public:
    explicit directory_entry(simple_fs::path p) : p_(std::move(p)) {}
    const simple_fs::path &path() const { return p_; }
    bool is_regular_file() const { return is_regular_file_path(p_); }
};

class recursive_directory_iterator {
    std::vector<directory_entry> entries_;

    void visit(const path &root) {
        DIR *dir = ::opendir(root.c_str());
        if (!dir) return;
        while (dirent *ent = ::readdir(dir)) {
            std::string name = ent->d_name;
            if (name == "." || name == "..") continue;
            path child = root / name;
            if (is_directory(child)) visit(child);
            else entries_.emplace_back(child);
        }
        ::closedir(dir);
    }

public:
    explicit recursive_directory_iterator(const path &root) { visit(root); }
    std::vector<directory_entry>::const_iterator begin() const { return entries_.begin(); }
    std::vector<directory_entry>::const_iterator end() const { return entries_.end(); }
};

} // namespace simple_fs
