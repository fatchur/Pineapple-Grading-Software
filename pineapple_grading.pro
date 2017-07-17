QT += core
QT -= gui

CONFIG += c++11

TARGET = pineapple_grading
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    detectpineapple.cpp

LIBS += `pkg-config --libs opencv`

HEADERS += \
    detectpineapple.h
