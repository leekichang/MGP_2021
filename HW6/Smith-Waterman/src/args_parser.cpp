#include <fstream>
#include <iostream>

#include "args_parser.h"



Parameters::Parameters(int argc_, char **argv_) {


    // default values
    sa = (1);
    sb = (4);
    gapo = (6);
    gape = (1);
    print_out = (0);
    query_batch_fasta_filename = "";
    target_batch_fasta_filename = "";

    argc = argc_;
    argv = argv_;

}

Parameters::~Parameters() {
}

void Parameters::print() {
    std::cerr <<  "[Arguments]" << std::endl;
    std::cerr <<  "sa=" << sa <<" , sb=" << sb <<" , gapo=" <<  gapo << " , gape="<<gape << std::endl;
    std::cerr <<  "print_out=" << print_out <<std::endl;
    std::cerr <<  "query_batch_fasta_filename=" << query_batch_fasta_filename <<" , target_batch_fasta_filename=" << target_batch_fasta_filename << std::endl;
}

void Parameters::failure(fail_type f) {
    switch(f)
    {
            case NOT_ENOUGH_ARGS:
                std::cerr << "Not enough Parameters. Required: -y AL_TYPE file1.fasta file2.fasta. See help (--help, -h) for usage. " << std::endl;
            break;
            case WRONG_ARG:
                std::cerr << "Wrong argument. See help (--help, -h) for usage. " << std::endl;
            break;
            case WRONG_FILES:
                std::cerr << "File error: either a file doesn't exist, or cannot be opened." << std::endl;
            break;

            default:
            break;
    }
    exit(1);
}

void Parameters::help() {
            std::cerr << "Usage: ./test_prog.out [-a] [-b] [-q] [-r] [-p] <query_batch.fasta> <target_batch.fasta>" << std::endl;
            std::cerr << "Options: -a INT    match score ["<< sa <<"]" << std::endl;
            std::cerr << "         -b INT    mismatch penalty [" << sb << "]"<< std::endl;
            std::cerr << "         -q INT    gap open penalty [" << gapo << "]" << std::endl;
            std::cerr << "         -r INT    gap extension penalty ["<< gape <<"]" << std::endl;
            std::cerr << "         -p        print the alignment results" << std::endl;
            std::cerr << "         --help, -h : displays this message." << std::endl;
            std::cerr << "		  "  << std::endl;
}


void Parameters::parse() {

    // before testing anything, check if calling for help.
    int c;
        
    std::string arg_next = "";
    std::string arg_cur = "";

    for (c = 1; c < argc; c++)
    {
        arg_cur = std::string((const char*) (*(argv + c) ) );
        arg_next = "";
        if (!arg_cur.compare("--help") || !arg_cur.compare("-h"))
        {
            help();
            exit(0);
        }
    }

    if (argc < 4)
    {
        failure(NOT_ENOUGH_ARGS);
    }

    for (c = 1; c < argc - 2; c++)
    {
        arg_cur = std::string((const char*) (*(argv + c) ) );
        if (arg_cur.at(0) == '-' && arg_cur.at(1) == '-' )
        {
            if (!arg_cur.compare("--help"))
            {
                help();
                exit(0);
            }

        } else if (arg_cur.at(0) == '-' )
        {
            if (arg_cur.length() > 2)
                failure(WRONG_ARG);
            char param = arg_cur.at(1);
            switch(param)
            {
              case 'a':
                c++;
                arg_next = std::string((const char*) (*(argv + c) ) );
                sa = std::stoi(arg_next);
                break;
              case 'b':
                c++;
                arg_next = std::string((const char*) (*(argv + c) ) );
                sb = std::stoi(arg_next);
                break;
              case 'q':
                c++;
                arg_next = std::string((const char*) (*(argv + c) ) );
                gapo = std::stoi(arg_next);
                break;
              case 'r':
                c++;
                arg_next = std::string((const char*) (*(argv + c) ) );
                gape = std::stoi(arg_next);
                break;
              case 'p':
                print_out = 1;
                break;
            }

            
        } else {
            failure(WRONG_ARG);
        }
    }


    // the last 2 Parameters are the 2 filenames.
    query_batch_fasta_filename = std::string( (const char*)  (*(argv + c) ) );
    c++;
    target_batch_fasta_filename = std::string( (const char*) (*(argv + c) ) );

}

