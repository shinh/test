#!perl -pl
s/\w+/$_{$&}.=$".$./eg}for(%_){
#for(;<>;){s/\w+/$_{$&}.=$".$./eg}print for%_

