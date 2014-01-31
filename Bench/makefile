
TARGET := Bench
INC := -I../include
LIBS := -lOpenCLIPP -lOpenCLIPP-C++
SRCEXT := cpp
SRCDIR := .
BUILDDIR := build

CC := g++
CFLAGS := -g -O3 -Wall -c -fmessage-length=0 -std=c++0x
LDFLAGS := -g -L..
RM := rm -rf

SOURCES := $(shell find $(SRCDIR) -type f -name '*.$(SRCEXT)')
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
DEPS := $(OBJECTS:.o=.deps)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo 'Building target: $@'
	@echo 'Invoking: Linker'
	$(CC) $(LDFLAGS) $^ -o $(TARGET) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

$(BUILDDIR)/src/%.o: $(SRCDIR)/src/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)/src
	@echo 'Building file: $<'
	$(CC) $(CFLAGS) $(INC) -MMD -MP -MF"$(@:.o=.deps)" -MT"$(@:.o=.deps)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

clean:
	@echo "Cleaning"
	$(RM) $(BUILDDIR) $(TARGET)
	@echo ' '

.PHONY: all clean
