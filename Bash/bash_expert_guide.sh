#!/bin/bash
# =============================================================================
# COMPLETE BASH EXPERT GUIDE
# From beginner to advanced bash scripting and command line mastery
# =============================================================================

# =============================================================================
# 1. BASH BASICS & FUNDAMENTALS
# =============================================================================

echo "=== BASH FUNDAMENTALS ==="

# Variables (no spaces around =)
name="John"
age=30
readonly PI=3.14159  # readonly variable

# Variable expansion
echo "Hello $name, you are $age years old"
echo "Hello ${name}, you are ${age} years old"  # Preferred for clarity

# Command substitution
current_date=$(date)
files_count=`ls | wc -l`  # Old style, avoid
echo "Today is $current_date"
echo "Files in directory: $files_count"

# Arithmetic
num1=10
num2=5
result=$((num1 + num2))
echo "10 + 5 = $result"

# Different arithmetic methods
result2=$(expr $num1 + $num2)
let result3=num1+num2
declare -i result4=num1+num2

# =============================================================================
# 2. ADVANCED VARIABLE TECHNIQUES
# =============================================================================

echo "=== ADVANCED VARIABLES ==="

# Parameter expansion
filename="/path/to/file.txt"
echo "Directory: ${filename%/*}"      # /path/to
echo "Filename: ${filename##*/}"      # file.txt
echo "Extension: ${filename##*.}"     # txt
echo "Name only: ${filename%.*}"      # /path/to/file

# Default values
echo "${undefined_var:-default_value}"        # Use default if unset
echo "${undefined_var:=assigned_default}"     # Assign and use default
echo "${undefined_var:?Error: variable unset}" # Error if unset

# String manipulation
text="Hello World"
echo "${text^}"          # Capitalize first letter
echo "${text^^}"         # Uppercase all
echo "${text,,}"         # Lowercase all
echo "${text:6}"         # Substring from position 6
echo "${text:0:5}"       # Substring: positions 0-5
echo "${text/World/Universe}"  # Replace first occurrence
echo "${text//o/0}"      # Replace all occurrences

# Arrays
declare -a fruits=("apple" "banana" "orange")
declare -A colors=([red]="#FF0000" [green]="#00FF00" [blue]="#0000FF")

# Array operations
fruits+=("grape")                    # Append
echo "All fruits: ${fruits[@]}"     # All elements
echo "First fruit: ${fruits[0]}"    # First element
echo "Number of fruits: ${#fruits[@]}" # Array length
echo "Red color: ${colors[red]}"     # Associative array

# =============================================================================
# 3. CONTROL STRUCTURES
# =============================================================================

echo "=== CONTROL STRUCTURES ==="

# If statements
score=85
if [[ $score -gt 90 ]]; then
    grade="A"
elif [[ $score -gt 80 ]]; then
    grade="B"
elif [[ $score -gt 70 ]]; then
    grade="C"
else
    grade="F"
fi
echo "Score: $score, Grade: $grade"

# Advanced conditionals
file="/etc/passwd"
if [[ -f "$file" && -r "$file" ]]; then
    echo "File exists and is readable"
fi

# String comparisons
if [[ "$USER" == "root" ]]; then
    echo "Running as root"
elif [[ "$USER" =~ ^[a-z]+$ ]]; then
    echo "Username contains only lowercase letters"
fi

# Case statements
case "$1" in
    start)
        echo "Starting service..."
        ;;
    stop)
        echo "Stopping service..."
        ;;
    restart)
        echo "Restarting service..."
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac

# Loops
echo "=== LOOPS ==="

# For loops
for i in {1..5}; do
    echo "Number: $i"
done

# For loop with arrays
for fruit in "${fruits[@]}"; do
    echo "I like $fruit"
done

# C-style for loop
for ((i=1; i<=3; i++)); do
    echo "Count: $i"
done

# While loop
counter=1
while [[ $counter -le 3 ]]; do
    echo "While counter: $counter"
    ((counter++))
done

# Until loop
counter=1
until [[ $counter -gt 3 ]]; do
    echo "Until counter: $counter"
    ((counter++))
done

# =============================================================================
# 4. FUNCTIONS
# =============================================================================

echo "=== FUNCTIONS ==="

# Basic function
greet() {
    echo "Hello, $1!"
}

# Function with multiple parameters
calculate() {
    local num1=$1
    local num2=$2
    local operation=$3
    
    case $operation in
        add) echo $((num1 + num2)) ;;
        sub) echo $((num1 - num2)) ;;
        mul) echo $((num1 * num2)) ;;
        div) 
            if [[ $num2 -eq 0 ]]; then
                echo "Error: Division by zero" >&2
                return 1
            fi
            echo $((num1 / num2))
            ;;
        *) 
            echo "Error: Unknown operation" >&2
            return 1
            ;;
    esac
}

# Function with return value
is_number() {
    [[ $1 =~ ^[0-9]+$ ]]
}

# Usage examples
greet "Alice"
result=$(calculate 10 5 add)
echo "10 + 5 = $result"

if is_number "123"; then
    echo "123 is a number"
fi

# =============================================================================
# 5. INPUT/OUTPUT & FILE OPERATIONS
# =============================================================================

echo "=== FILE OPERATIONS ==="

# Reading user input
read -p "Enter your name: " username
read -s -p "Enter password: " password  # Silent input
echo # New line after silent input

# Reading from files
if [[ -f "/etc/passwd" ]]; then
    # Read line by line
    while IFS=: read -r username x uid gid gecos home shell; do
        [[ $uid -ge 1000 ]] && echo "Regular user: $username ($home)"
    done < /etc/passwd | head -5
fi

# Writing to files
cat > temp_file.txt << 'EOF'
This is a temporary file
with multiple lines
created using here document
EOF

# File tests
test_file="temp_file.txt"
[[ -f "$test_file" ]] && echo "File exists"
[[ -r "$test_file" ]] && echo "File is readable"
[[ -w "$test_file" ]] && echo "File is writable"
[[ -x "$test_file" ]] && echo "File is executable"
[[ -s "$test_file" ]] && echo "File is not empty"

# File operations
cp "$test_file" "${test_file}.backup"
chmod 644 "$test_file"
stat "$test_file"

# =============================================================================
# 6. ADVANCED BASH FEATURES
# =============================================================================

echo "=== ADVANCED FEATURES ==="

# Process substitution
echo "Files in current directory:"
diff <(ls /tmp) <(ls /var/tmp) || echo "Directories differ"

# Brace expansion
echo "Backup files:"
touch file{1..3}.txt
ls file*.txt
mkdir -p project/{src,bin,doc,test}

# Command grouping
{
    echo "Grouped commands"
    date
    whoami
} > grouped_output.txt

# Subshells
(
    cd /tmp
    echo "In subshell, PWD: $PWD"
)
echo "Back in main shell, PWD: $PWD"

# =============================================================================
# 7. ERROR HANDLING & DEBUGGING
# =============================================================================

echo "=== ERROR HANDLING ==="

# Exit on error
set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Exit if using undefined variable
set -o pipefail  # Pipeline fails if any command fails

# Error handling function
handle_error() {
    echo "Error on line $1" >&2
    exit 1
}

# Set error trap
trap 'handle_error $LINENO' ERR

# Safe file operations
backup_file() {
    local source="$1"
    local backup="${source}.backup"
    
    if [[ ! -f "$source" ]]; then
        echo "Error: Source file '$source' does not exist" >&2
        return 1
    fi
    
    if ! cp "$source" "$backup" 2>/dev/null; then
        echo "Error: Failed to create backup" >&2
        return 1
    fi
    
    echo "Backup created: $backup"
}

# Debugging techniques
set -x  # Print commands before executing (debug mode)
# Your code here
set +x  # Disable debug mode

# =============================================================================
# 8. TEXT PROCESSING MASTERY
# =============================================================================

echo "=== TEXT PROCESSING ==="

# Create sample data
cat > sample.csv << 'EOF'
name,age,city,salary
John,30,New York,75000
Alice,25,London,65000
Bob,35,Paris,80000
Carol,28,Tokyo,70000
EOF

# AWK examples
echo "=== AWK Processing ==="
awk -F',' 'NR>1 {print $1 " is " $2 " years old"}' sample.csv
awk -F',' 'NR>1 {sum+=$4; count++} END {print "Average salary:", sum/count}' sample.csv

# SED examples
echo "=== SED Processing ==="
sed 's/,/ | /g' sample.csv  # Replace commas with pipes
sed -n '2,3p' sample.csv     # Print lines 2-3
sed '1d' sample.csv | sed 's/\([^,]*\),\([^,]*\),.*/\2 \1/' # Extract and rearrange

# GREP examples
echo "=== GREP Processing ==="
grep -E '^[A-C]' sample.csv                    # Names starting with A-C
grep -v '^name' sample.csv | grep -E ',[67][0-9][0-9][0-9][0-9]$'  # Salaries 60k-79k

# Complex text processing pipeline
echo "=== Complex Pipeline ==="
cat sample.csv | \
    grep -v '^name' | \
    awk -F',' '{print $3 "," $4}' | \
    sort -t',' -k2 -nr | \
    head -2

# =============================================================================
# 9. SYSTEM ADMINISTRATION TASKS
# =============================================================================

echo "=== SYSTEM ADMINISTRATION ==="

# Process management
ps_info() {
    echo "=== Process Information ==="
    ps aux | head -10
    echo "Total processes: $(ps aux | wc -l)"
    echo "Memory usage: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
}

# Disk usage monitoring
disk_usage() {
    echo "=== Disk Usage ==="
    df -h | grep -E '^/dev/'
    echo "Largest directories in /var/log:"
    du -sh /var/log/* 2>/dev/null | sort -hr | head -5
}

# Log analysis
analyze_logs() {
    local logfile="/var/log/syslog"
    if [[ -r "$logfile" ]]; then
        echo "=== Log Analysis ==="
        echo "Recent errors:"
        grep -i error "$logfile" | tail -5
        echo "Most frequent log sources:"
        awk '{print $5}' "$logfile" | sort | uniq -c | sort -nr | head -5
    fi
}

# Network information
network_info() {
    echo "=== Network Information ==="
    echo "Active connections:"
    netstat -tuln | grep LISTEN | head -10
    echo "Network interfaces:"
    ip addr show | grep -E '^[0-9]+:' | cut -d: -f2
}

# =============================================================================
# 10. PERFORMANCE & OPTIMIZATION
# =============================================================================

echo "=== PERFORMANCE OPTIMIZATION ==="

# Timing operations
time_operation() {
    echo "Timing different operations:"
    
    # Method 1: External command
    time1=$({ time for i in {1..1000}; do echo $i >/dev/null; done; } 2>&1 | grep real | awk '{print $2}')
    
    # Method 2: Built-in timing
    start_time=$(date +%s.%N)
    for i in {1..1000}; do
        : # null command
    done
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "calculation error")
    
    echo "Loop with echo: $time1"
    echo "Empty loop: ${elapsed}s"
}

# Memory-efficient processing
process_large_file() {
    local filename="$1"
    
    # Bad: loads entire file into memory
    # content=$(cat "$filename")
    
    # Good: process line by line
    while IFS= read -r line; do
        # Process line
        echo "Processing: ${line:0:50}..."
    done < "$filename"
}

# Parallel processing
parallel_task() {
    echo "=== Parallel Processing ==="
    
    # Sequential
    echo "Sequential processing:"
    time {
        for i in {1..4}; do
            sleep 1
            echo "Task $i completed"
        done
    }
    
    # Parallel
    echo "Parallel processing:"
    time {
        for i in {1..4}; do
            (sleep 1; echo "Parallel task $i completed") &
        done
        wait  # Wait for all background jobs
    }
}

# =============================================================================
# 11. SECURITY & BEST PRACTICES
# =============================================================================

echo "=== SECURITY BEST PRACTICES ==="

# Secure temp files
create_temp_file() {
    local temp_file
    temp_file=$(mktemp) || {
        echo "Failed to create temp file" >&2
        return 1
    }
    
    # Set restrictive permissions
    chmod 600 "$temp_file"
    
    echo "Created secure temp file: $temp_file"
    
    # Cleanup trap
    trap "rm -f '$temp_file'" EXIT
}

# Input validation
validate_email() {
    local email="$1"
    local pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    if [[ $email =~ $pattern ]]; then
        echo "Valid email: $email"
        return 0
    else
        echo "Invalid email: $email" >&2
        return 1
    fi
}

# Safe command execution
safe_execute() {
    local cmd="$1"
    
    # Log the command
    echo "Executing: $cmd" | logger -t "safe_execute"
    
    # Execute with timeout
    if timeout 30s bash -c "$cmd"; then
        echo "Command executed successfully"
    else
        echo "Command failed or timed out" >&2
        return 1
    fi
}

# Password generation
generate_password() {
    local length=${1:-16}
    
    # Method 1: Using /dev/urandom
    tr -dc 'A-Za-z0-9!@#$%^&*' < /dev/urandom | head -c "$length"
    echo
    
    # Method 2: Using openssl
    openssl rand -base64 "$length" 2>/dev/null | head -c "$length"
    echo
}

# =============================================================================
# 12. TESTING & QUALITY ASSURANCE
# =============================================================================

echo "=== TESTING FRAMEWORK ==="

# Simple test framework
TEST_COUNT=0
PASS_COUNT=0

assert_equals() {
    local expected="$1"
    local actual="$2"
    local description="$3"
    
    ((TEST_COUNT++))
    
    if [[ "$expected" == "$actual" ]]; then
        echo "✓ PASS: $description"
        ((PASS_COUNT++))
    else
        echo "✗ FAIL: $description"
        echo "  Expected: '$expected'"
        echo "  Actual:   '$actual'"
    fi
}

assert_file_exists() {
    local filename="$1"
    local description="$2"
    
    ((TEST_COUNT++))
    
    if [[ -f "$filename" ]]; then
        echo "✓ PASS: $description"
        ((PASS_COUNT++))
    else
        echo "✗ FAIL: $description - File '$filename' does not exist"
    fi
}

# Test examples
run_tests() {
    echo "Running tests..."
    
    # Test arithmetic
    result=$((5 + 3))
    assert_equals "8" "$result" "Addition test"
    
    # Test string manipulation
    text="Hello World"
    result="${text// /_}"
    assert_equals "Hello_World" "$result" "String replacement test"
    
    # Test file operations
    touch test_file.tmp
    assert_file_exists "test_file.tmp" "File creation test"
    rm -f test_file.tmp
    
    echo "Tests completed: $PASS_COUNT/$TEST_COUNT passed"
}

# =============================================================================
# 13. REAL-WORLD EXAMPLES
# =============================================================================

echo "=== REAL-WORLD EXAMPLES ==="

# System backup script
system_backup() {
    local backup_dir="/backup/$(date +%Y%m%d)"
    local config_files=("/etc/hosts" "/etc/fstab" "/etc/crontab")
    
    echo "Starting system backup to $backup_dir"
    
    # Create backup directory
    mkdir -p "$backup_dir" || {
        echo "Failed to create backup directory" >&2
        return 1
    }
    
    # Backup configuration files
    for file in "${config_files[@]}"; do
        if [[ -f "$file" ]]; then
            cp "$file" "$backup_dir/"
            echo "Backed up: $file"
        fi
    done
    
    # Create tarball
    tar -czf "${backup_dir}.tar.gz" -C "$(dirname "$backup_dir")" "$(basename "$backup_dir")"
    rm -rf "$backup_dir"
    
    echo "Backup completed: ${backup_dir}.tar.gz"
}

# Log rotation script
rotate_logs() {
    local log_dir="/var/log/myapp"
    local max_age=7  # days
    
    if [[ ! -d "$log_dir" ]]; then
        echo "Log directory does not exist: $log_dir" >&2
        return 1
    fi
    
    echo "Rotating logs in $log_dir"
    
    # Compress old logs
    find "$log_dir" -name "*.log" -mtime +1 -exec gzip {} \;
    
    # Remove old compressed logs
    find "$log_dir" -name "*.log.gz" -mtime +"$max_age" -delete
    
    echo "Log rotation completed"
}

# Service monitor
monitor_service() {
    local service_name="$1"
    local max_attempts=3
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if systemctl is-active "$service_name" >/dev/null 2>&1; then
            echo "$service_name is running"
            return 0
        else
            echo "$service_name is not running, attempting restart..."
            systemctl restart "$service_name"
            sleep 5
            ((attempt++))
        fi
    done
    
    echo "Failed to start $service_name after $max_attempts attempts" >&2
    return 1
}

# =============================================================================
# 14. ADVANCED TIPS & TRICKS
# =============================================================================

echo "=== ADVANCED TIPS & TRICKS ==="

# Useful aliases and functions
useful_aliases() {
    cat << 'EOF'
# Add these to your ~/.bashrc

# Navigation
alias ..='cd ..'
alias ...='cd ../..'
alias l='ls -la'
alias ll='ls -alF'
alias la='ls -A'

# Safety
alias cp='cp -i'
alias mv='mv -i'
alias rm='rm -i'

# Utilities
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'
alias df='df -h'
alias du='du -h'
alias free='free -h'

# Functions
mkcd() { mkdir -p "$1" && cd "$1"; }
extract() {
    if [[ -f "$1" ]]; then
        case "$1" in
            *.tar.bz2)   tar xjf "$1"     ;;
            *.tar.gz)    tar xzf "$1"     ;;
            *.bz2)       bunzip2 "$1"     ;;
            *.rar)       unrar x "$1"     ;;
            *.gz)        gunzip "$1"      ;;
            *.tar)       tar xf "$1"      ;;
            *.tbz2)      tar xjf "$1"     ;;
            *.tgz)       tar xzf "$1"     ;;
            *.zip)       unzip "$1"       ;;
            *.Z)         uncompress "$1"  ;;
            *.7z)        7z x "$1"        ;;
            *)           echo "Cannot extract '$1'" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}
EOF
}

# Prompt customization
custom_prompt() {
    cat << 'EOF'
# Add to ~/.bashrc for a custom prompt

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
NC='\033[0m' # No Color

# Git branch function
parse_git_branch() {
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}

# Custom prompt
PS1="\[${GREEN}\]\u@\h\[${NC}\]:\[${BLUE}\]\w\[${YELLOW}\]\$(parse_git_branch)\[${NC}\]\$ "
EOF
}

# =============================================================================
# 15. SUMMARY & CHEAT SHEET
# =============================================================================

echo "=== BASH EXPERT CHEAT SHEET ==="

cat << 'EOF'
BASH EXPERT CHEAT SHEET
======================

VARIABLES & EXPANSION:
  var="value"                    # Assign variable
  $var or ${var}                 # Variable expansion
  ${var:-default}                # Default value
  ${var/old/new}                 # Replace first occurrence
  ${var//old/new}                # Replace all occurrences
  ${#var}                        # String length

CONDITIONALS:
  [[ condition ]]                # Modern test
  [[ -f file ]]                  # File exists
  [[ -d dir ]]                   # Directory exists
  [[ string =~ regex ]]          # Regex match
  [[ $a -eq $b ]]                # Numeric equality
  [[ $a == $b ]]                 # String equality

LOOPS:
  for i in {1..10}; do ... done  # Range loop
  for file in *.txt; do ... done # Glob loop
  while condition; do ... done   # While loop
  until condition; do ... done   # Until loop

FUNCTIONS:
  func() { commands; }           # Function definition
  local var=value                # Local variable
  return 0                       # Return status

TEXT PROCESSING:
  grep pattern file              # Search text
  sed 's/old/new/g' file        # Replace text
  awk '{print $1}' file         # Print column
  cut -d: -f1 file              # Cut fields
  sort file                      # Sort lines
  uniq file                      # Remove duplicates

I/O REDIRECTION:
  command > file                 # Redirect stdout
  command 2> file                # Redirect stderr
  command &> file                # Redirect both
  command < file                 # Input from file
  command | command2             # Pipe
  command1 && command2           # AND operator
  command1 || command2           # OR operator

SPECIAL VARIABLES:
  $0                            # Script name
  $1, $2, ...                   # Arguments
  $@                            # All arguments
  $#                            # Argument count
  $$                            # Process ID
  $?                            # Exit status
  $!                            # Last background PID

DEBUGGING:
  set -x                        # Debug mode
  set -e                        # Exit on error
  set -u                        # Exit on undefined
  bash -x script.sh             # Debug script

SHORTCUTS:
  Ctrl+A                        # Beginning of line
  Ctrl+E                        # End of line
  Ctrl+L                        # Clear screen
  Ctrl+R                        # Search history
  !!                            # Last command
  !$                            # Last argument
EOF

echo ""
echo "🎉 Congratulations! You now have expert-level bash knowledge."
echo "Practice these concepts regularly to master bash scripting."
echo ""
echo "Next steps:"
echo "1. Practice writing complex scripts"
echo "2. Study system administration tasks"
echo "3. Learn advanced tools like jq, parallel, etc."
echo "4. Contribute to open source bash projects"

# Cleanup
rm -f temp_file.txt temp_file.txt.backup grouped_output.txt sample.csv test_file.tmp

echo ""
echo "Guide completed successfully! 🚀"