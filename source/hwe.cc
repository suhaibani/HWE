#include <iostream>
#include <fstream>
#include <cstdio>
#include <utility>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>
#include <vector>
#include <map>
#include <set>
#include<unordered_map>
#include <Eigen/Dense>

#include "parse_args.hh"
#include "omp.h"

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

using namespace std;
using namespace boost;
using namespace Eigen;

map <string,vector<string>> relations;

int w1,w2;

int D; // dimensionality

unordered_map<string, VectorXd> w; // target word vectors
unordered_map<string, VectorXd> c; // context word vectors
unordered_map<string, double> bw; // target word biases
unordered_map<string, double> bc; // context word biases

unordered_map<string, VectorXd> grad_w; // Squared gradient for AdaGrad
unordered_map<string, VectorXd> grad_c; 
unordered_map<string, double> grad_bw; 
unordered_map<string, double> grad_bc; 

set<string> words;
set<string> concepts;

struct edge {
  string c;
  double value;
};

vector<edge> edges;
map <string,vector<edge>> co_occurrences;
string clean_string(string temp)
{
 std::replace(temp.begin(),temp.end(),'[',' ');
 std::replace(temp.begin(),temp.end(),']',' ');
 std::replace(temp.begin(),temp.end(),'\'',' ');
 temp.erase(std::remove(temp.begin(), temp.end(), ' '), temp.end());
 
 return temp;

}

int insert_relations(string word,string rel)
{

	boost::char_separator<char> sep(",");
	typedef boost::tokenizer< boost::char_separator<char> > t_tokenizer;
	t_tokenizer tok(rel, sep);
	std::vector<string> temp;
	if(word.length()>=2)
	{
		if(relations.find(word)!=relations.end()){
			temp=relations[word];
		
		}
		for (t_tokenizer::iterator beg = tok.begin(); beg != tok.end(); ++beg){
    		string t =*beg;
    		temp.push_back(t);
    		concepts.insert(t);
  	 	}
		// update relations
		if(relations.find(word)!=relations.end()){	
			relations[word]=temp;
		}
		// insert new realtion
		else{
			words.insert(word);
			relations.insert(std::make_pair(word,temp));
		}
	
	}


}


void test_map()
{
	ofstream output;output.open("check_map.txt");
	for (auto i=relations.begin();i!=relations.end();++i){
		output<<"word  - "<<i->first<<endl;
		w2++;
		vector<string> temp=i->second;
		//cout<<"Relations"<<endl;
		//output<<"	";
		int z=0;
		for (auto j=temp.begin();j!=temp.end();++j){
			output<<*j;
			z++;
		}
        output<<endl;
	}
	cout<<"Word Count from file..."<<w1<<endl;
	cout<<"Word Count from dic...."<<w2<<endl;
	output.close();
}


void read_file(string fname)
{

    ifstream train_file(fname, std::ifstream::in);
    string line, relations,word;
    string prev;
    double value;
    cout<<"Reading file ..."<<fname<<endl;

    while (!train_file.eof()){
       	getline(train_file,line);
    	boost::trim(line);
    	if(line.length()>1){
    	// complete relation
    		if((line.find('[')!=string::npos) && (line.find(']')!=string::npos)){
				line=clean_string(line);
				insert_relations(word,line);
				prev="";word="";
			}// relation first part
    		else if ((line.find('[')!=string::npos) && (line.find(']')==string::npos)){
				prev=line;
			}
			// relation mid portions
			else if ((line.find('[')==string::npos) && (line.find(']')==string::npos) && (line.find(',')!=string::npos)){
			prev+=line;
			}
			// relation last part 
			else if ((line.find('[')==string::npos)&& (line.find(']')!=string::npos)){
				prev+=line;
				prev=clean_string(prev);
				//cout<<" relation - "<<prev<<endl;
				insert_relations(word,line);
				prev="";
				word="";
			
			}
			else{
				//line=clean_string(line);
				word=line;
				w1++;
			}
    	}
	}

    train_file.close();
    cout<<"Loaded Relations in memory..."<<endl;
    // allocate memory for all vectors
}  

double f(size_t x){
    if (x < 100)
        return pow((x / 100.0), 0.75);
    else
        return 1.0;
}


void train(int epohs,double alpha, double lambda)
{
	fprintf(stderr, "%s\nTotal ephos to train = %d\n%s", KGRN, epohs, KNRM);
    fprintf(stderr, "%sInitial learning rate = %f\n%s", KGRN, alpha, KNRM);
    fprintf(stderr, "%slambda = %f\n%s", KGRN, lambda, KNRM);
    fprintf(stderr, "%sDim = %d\n%s", KGRN, D, KNRM);
    
    ofstream output;output.open("equation_test.txt");
    double total_loss, cost_corp,cost_lex;
    
    VectorXd gw = VectorXd::Zero(D);
    VectorXd gc = VectorXd::Zero(D);  
    VectorXd diff = VectorXd::Zero(D); 

    VectorXd one_vect = VectorXd::Zero(D);
    std::vector<string> temp;
    
    for (auto i = 0; i < D; ++i)
        one_vect[i] = 1.0;

    int found_pairs = 0;
    int counter=0;
    double rel_value=0;
    int cooccurr_val=0;
    vector<edge> t2;
    int x_val=0;
    bool corpus_rel=false;
    string r,co_occ,conc;
    

    for (int t = 0; t < epohs; ++t){
        total_loss = 0;
        found_pairs = 0;
        

        // for all words in the lexicon
        for(auto keys = relations.begin(); keys != relations.end(); ++keys){
        	counter=0;
        	string word=keys->first;
        	temp=keys->second;
        	int length=temp.size();
        	// for all concepts associated with a word
        	for (auto rel=temp.begin();rel!=temp.end();++rel){  
                // rel_value is lambda
             	rel_value=length-counter;
             	counter++;
           	    x_val=0;
             	r=*rel; conc=*rel;
                corpus_rel=false;  x_val=0;
                co_occ="";
             	// search for word, concept pair in the co-occurrence matrix
             	if(co_occurrences.find(word)!=co_occurrences.end()){
             		t2=co_occurrences[word];
             		for(auto occurr=t2.begin();occurr!=t2.end();++occurr){
             			edge e=*occurr;
             			if (*rel==e.c){
             				x_val=e.value;
             				co_occ=e.c;
             				corpus_rel=true;
                            break;
             				/* code */
             			}

             		}
             	}

            if(corpus_rel){
            
            //   output<<"CP - "<<word<<" - concept order - "<<rel_value<<" Concept "<<*rel<<" + from corpus "<<word<<" , "<<co_occ<< " co-occurrence value "<<X_val<<endl;
                cost_corp = w[word].dot(c[co_occ]) + bw[word] + bc[co_occ] - log(x_val);
                cost_corp *= f(x_val);
            }
            else{
                //cost_lex = (w[word] - (rel_value * c[conc].dot(c[conc]))).squaredNorm();
                cost_corp=0;
                 //output<<"CAbsent - "<<word<<" - concept order "<<rel_value<<" Concept "<<*rel<<"  from corpus - no matching instances "<<endl;
            }
            
            //total_loss += f(x_val) * cost_corp * cost_corp;

          
            gw = (w[word] - ((rel_value * 0.1) * c[conc]))+(cost_corp * c[conc]) ;
            gc = (w[word] - ((rel_value * 0.1) * c[conc] * ( -1 * (rel_value * 0.1))))+(cost_corp * w[word]);

            grad_w[word] += gw.cwiseProduct(gw);
            grad_c[conc] += gc.cwiseProduct(gc);
            grad_bw[word] += cost_corp * cost_corp;
            grad_bc[conc] += cost_corp * cost_corp;

            w[word] -= alpha * gw.cwiseProduct((grad_w[word] + one_vect).cwiseInverse().cwiseSqrt());
            c[conc] -= alpha * gc.cwiseProduct((grad_c[conc] + one_vect).cwiseInverse().cwiseSqrt());

            bw[word] -= (alpha * cost_corp) / sqrt(1.0 + grad_bw[word]);
            bc[conc] -= (alpha * cost_corp) / sqrt(1.0 + grad_bc[conc]);

           
            }

        }
        //fprintf(stderr, "Itr = %d, Loss = %f, foundPairs = %d\n", t, (sqrt(total_loss) / edges.size()), found_pairs);
     //output.close();
        fprintf(stderr, "Itr = %d\n", t);
     output.close();    
    }
}
void load_cooccurrences(string fname,vector<edge>& train_data)
{
	cout<<"Loading Co-occurrences..."<<endl;
    ifstream train_file(fname.c_str());

    string first, second;
    double value;
    int discard_count=0;
    int count=0;
    //ofstream output;output.open("discarded_cooccurrences.txt");
    
    vector<string> temp_rel;
    
    while (train_file >> first >> second >> value){
    	edge e;
        vector<edge> temp;
        if(relations.find(first)!=relations.end()){	
            temp_rel=relations[first];
        	std::vector<string>::iterator it;
			it = find (temp_rel.begin(), temp_rel.end(), second);
        	
            if(it!=temp_rel.end()){	
        			e.c=second;
        			e.value=value;
        			//cout<<"Relation found in lexicon.."<<first<<" ->"<<*it<<endl;
        			if(co_occurrences.find(first)!=co_occurrences.end()){
        				temp=co_occurrences[first];
        				temp.push_back(e);
        				co_occurrences[first]=temp;

        			}
        			else{
        				temp.push_back(e);
        				co_occurrences.insert(std::make_pair(first,temp));
        			}
        			
        			count++;
        			
        		}
        

        }
       

  
    }
    
    train_file.close();
  //  output.close();
    // allocate memory for all vectors
}

void centralize(unordered_map<string, VectorXd> &x){
    VectorXd mean = VectorXd::Zero(D);
    VectorXd squared_mean = VectorXd::Zero(D);
    for (auto w = x.begin(); w != x.end(); ++w){
        mean += w->second;
        squared_mean += (w->second).cwiseProduct(w->second);
    }
    mean = mean / ((double) x.size());
    VectorXd sd = squared_mean - mean.cwiseProduct(mean);
    for (int i = 0; i < D; ++i){
        sd[i] = sqrt(sd[i]);
    }
    for (auto w = x.begin(); w != x.end(); ++w){
        VectorXd tmp = VectorXd::Zero(D);
        for (int i = 0; i < D; ++i){
            tmp[i] = (w->second)[i] - mean[i];
            if (sd[i] != 0)
                tmp[i] /= sd[i];
        }
        w->second = tmp;
    }
}

void initialize(){
    int count_words = 0;
    for(auto e = words.begin(); e != words.end(); ++e){
        count_words++;
        w[*e] = VectorXd::Random(D);
        bw[*e] = 0;
        grad_w[*e] = VectorXd::Zero(D);
        grad_bw[*e] = 0;
    }

    int count_contexts = 0;
    for(auto e = concepts.begin(); e != concepts.end(); ++e){
        count_contexts++;
        c[*e] = VectorXd::Random(D);
        bc[*e] = 0;
        grad_c[*e] = VectorXd::Zero(D);
        grad_bc[*e] = 0;
    }

    centralize(w);
    centralize(c);
    cout<<"Initialization Completed..."<<endl;
}

void write_line(ofstream &reps_file, VectorXd vec, string label){
    reps_file << label + " ";
    for (int i = 0; i < D; ++i)
        reps_file << vec[i] << " ";
    reps_file << endl;
}

void save_model(string fname){
    ofstream reps_file;
    reps_file.open(fname);
    if (!reps_file){
        cout<<"Failed to write reps to "<< fname.c_str()<<endl;
        exit(1);
    } 
    for (auto x = words.begin(); x != words.end(); ++x){
        if (concepts.find(*x) != concepts.end())
             //write_line(reps_file, 0.5 * (w[*x] + c[*x]), *x);
             write_line(reps_file, 1 * (w[*x]), *x);
        else
            write_line(reps_file, w[*x], *x);     
    }
    ofstream context;
    context.open("vectors_hwe_hypo");
    for (auto x = concepts.begin(); x != concepts.end(); ++x){
        //if (words.find(*x) == words.end())
            write_line(context, c[*x], *x);
    }
    reps_file.close();
}

int main(int argc, char *argv[])
{
	/* 
    D = 300;
    int epohs = 1;
    double alpha = 0.5;
    string model = "";
    double lambda = 0;
    */
    int no_threads = 100;
   // omp_set_num_threads(no_threads);
    //setNbThreads(no_threads);
    //initParallel(); 

    if (argc == 1) {
        fprintf(stderr, "usage: ./reps --dim=dimensionality --model=model_fname \
                                --alpha=alpha --ephos=rounds --lmda=lambda --edges=edges_fname \
                                 --lex=lexicon_file_name \n"); 
        return 0;
    }
    parse_args::init(argc, argv); 
    string edges_fname = parse_args::get<string>("--corpus");
    string lexicon_fname = parse_args::get<string>("--lex");

    D = parse_args::get<int>("--dim");
    int epohs = parse_args::get<int>("--epohs");
    double alpha = parse_args::get<double>("--alpha");
    string model = parse_args::get<string>("--output");
    double lambda = parse_args::get<double>("--lambda");
    
    read_file(lexicon_fname);
    load_cooccurrences(edges_fname,edges);
    
    fprintf(stderr, "%sTotal no of words = %d\n%s", KGRN, (int) words.size(), KNRM);
    fprintf(stderr, "%sTotal no. of contexts = %d\n%s", KGRN, (int) concepts.size(), KNRM); 
    
    initialize();
    train(epohs, alpha, lambda);
    save_model(model);

    return 0;

//	test_map();
}
