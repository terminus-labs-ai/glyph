package main

// MyStruct is a sample struct.
// It has multiple fields.
type MyStruct struct {
	Name string
	age  int
}

// NewMyStruct creates a new instance.
func NewMyStruct(name string, age int) *MyStruct {
	return &MyStruct{Name: name, age: age}
}

// GetName returns the name.
func (s *MyStruct) GetName() string {
	return s.Name
}

// setAge sets the age (unexported).
func (s MyStruct) setAge(age int) {
	s.age = age
}

// Stringer is an interface.
type Stringer interface {
	String() string
}

// standalone function
func greet(name string) string {
	return "Hello, " + name
}

// multiReturn demonstrates multiple returns.
func multiReturn(x int) (int, error) {
	return x * 2, nil
}

// MaxSize is the maximum size.
const MaxSize = 1024

const (
	StatusActive   = "active"
	StatusInactive = "inactive"
)

// DefaultName is the default name.
var DefaultName = "world"

var (
	counter int
	enabled bool
)

// StringList is a type alias.
type StringList = []string

// UserID is a named type.
type UserID int64
