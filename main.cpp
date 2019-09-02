#include "mainwindow.h"
#include "sim_2d.h"
#include "arucomatcher2d.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    SIM_2D *sim = new ArucoMatcher2D();

    sim->run();

    // Take quaternion;


    // Send to pipe.



    return a.exec();
}
