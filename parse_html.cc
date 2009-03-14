#include <stdio.h>

#include <string>

#include <Qt/qapplication.h>
#include <dom/html_document.h>
#include <dom/dom_string.h>
#include <dom/dom_doc.h>
#include <kjs/kjsobject.h>
#include <kjs/kjsinterpreter.h>

int main(int argc, char* argv[]) {
    QApplication a(argc, argv);

    KJSInterpreter kjs;
    KJSO(&doc);
    //KJSGlobalObject obj = 

    DOM::HTMLDocument doc;
    doc.open();
    doc.write("<!doctype html><html><head><title>hoge</title></head><body></body></html>");
    printf("%s\n", doc.title().string().toAscii().data());
}
