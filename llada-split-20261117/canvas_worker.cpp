// Research-only canvas adapter, pinned to the accompanying native build.
// This is not a shipped ABI or Skippy wire-protocol implementation.
#include "skippy.h"
#include "gguf.h"
#include "skippy/runtime_types.h"
#include "skippy/activation_internal.h"
#include "skippy/graph_build_inputs.h"
#include "vendor/nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
static constexpr size_t MAX_TOKENS=3072, MAX_FRAME=64*1024*1024;
static skippy_error * error=nullptr;
static void check(skippy_status status,const char * label) {
    if(status!=SKIPPY_STATUS_OK) throw std::runtime_error(std::string(label)+": "+(error?error->message:"native failure"));
}
static std::string plan_string(skippy_stage_plan * plan,skippy_stage_plan_string_ref_v1 ref) {
    const char * data=nullptr;size_t n=0;
    check(skippy_stage_plan_string_v1(plan,ref,&data,&n,&error),"plan string");return {data,n};
}
static std::vector<const char *> pointers(const std::vector<std::string> & strings) {
    std::vector<const char *> p;for(const auto & s:strings)p.push_back(s.c_str());return p;
}
static skippy_stage_plan * make_plan(const char * path,int start,int end) {
    auto * meta=gguf_init_from_file(path,{true,nullptr});
    if(!meta) throw std::runtime_error("GGUF metadata failed");
    std::vector<skippy_stage_planner_tensor_v1> tensors;
    for(int64_t i=0;i<gguf_get_n_tensors(meta);++i) {
        skippy_stage_planner_tensor_v1 t{};t.abi_version=1;t.struct_size=sizeof(t);
        t.tensor_id=t.native_name=gguf_get_tensor_name(meta,i);t.ggml_type=gguf_get_tensor_type(meta,i);
        const auto * dims=gguf_get_tensor_ne(meta,i);t.dimension_count=4;
        while(t.dimension_count>1 && dims[t.dimension_count-1]==1)--t.dimension_count;
        std::copy(dims,dims+4,t.dimensions);
        t.data_offset=gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,i);t.stored_length=gguf_get_tensor_size(meta,i);
        tensors.push_back(t);
    }
    std::sort(tensors.begin(),tensors.end(),[](const auto & a,const auto & b){return std::strcmp(a.tensor_id,b.tensor_id)<0;});
    skippy_stage_planner_profile_v1 profiles[]={
        {1,sizeof(skippy_stage_planner_profile_v1),"canvas",512,1,512,0},
        {1,sizeof(skippy_stage_planner_profile_v1),"long",2304,1,2304,0},
        {1,sizeof(skippy_stage_planner_profile_v1),"small",37,1,37,0}};
    skippy_stage_planner_config_v1 config{1,sizeof(config),"sha256:8d313697ff01d76e0477ec30fe0b117e816d4da63af091b4d6a2a3e2eba23355",
        &path,1,tensors.data(),tensors.size(),profiles,3,"canvas-ctx3072-no-cache","metal"};
    skippy_stage_planner * planner=nullptr;skippy_stage_plan * plans[2]{};
    check(skippy_stage_planner_create_v1(&config,&planner,&error),"planner");
    check(skippy_stage_planner_realize_v1(planner,0,16,&plans[0],&error),"head plan");
    check(skippy_stage_planner_realize_v1(planner,16,32,&plans[1],&error),"tail plan");
    const skippy_stage_plan * chain[]={plans[0],plans[1]};
    check(skippy_stage_plan_validate_chain_v1(chain,2,&error),"chain validation");
    auto * selected=plans[start==0?0:1];skippy_stage_plan_free(plans[start==0?1:0]);
    skippy_stage_planner_free(planner);gguf_free(meta);
    (void)end;return selected;
}
static skippy_model * open_model(const char * path,const std::string & mode) {
    auto config=skippy_runtime_config_default();
    config.layer_start=mode=="tail"?16:0;config.layer_end=mode=="head"?16:32;config.stage_index=mode=="tail"?1:0;
    config.ctx_size=MAX_TOKENS;config.n_batch=MAX_TOKENS;config.n_ubatch=MAX_TOKENS;
    config.lane_count=1;config.n_gpu_layers=999;config.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    std::string contract;std::vector<std::string> resident,ii,ib,ei,eb;
    skippy_stage_plan * plan=nullptr;
    if(mode!="full") {
        plan=make_plan(path,config.layer_start,config.layer_end);
        skippy_stage_plan_desc_v1 d{};check(skippy_stage_plan_describe_v1(plan,&d,&error),"describe");
        contract=plan_string(plan,d.execution_contract);
        for(size_t i=0;i<d.resident_tensor_count;++i) {
            skippy_stage_plan_value_desc_v1 v{};check(skippy_stage_plan_resident_tensor_at_v1(plan,i,&v,&error),"resident");
            resident.push_back(plan_string(plan,v.identity));
        }
        for(size_t p=0;p<d.profile_count;++p) {
            skippy_stage_plan_profile_desc_v1 desc{};check(skippy_stage_plan_profile_at_v1(plan,p,&desc,&error),"profile");
            std::vector<std::string> pi,pib,pe,peb;
            for(int kind=0;kind<2;++kind) {
                auto & ids=kind==0?pi:pe;auto & binds=kind==0?pib:peb;
                size_t n=kind==0?desc.activation_import_count:desc.activation_export_count;
                for(size_t i=0;i<n;++i) {
                    skippy_stage_plan_value_desc_v1 v{};
                    check(skippy_stage_plan_value_at_v1(plan,p,static_cast<skippy_stage_plan_value_kind>(kind),i,&v,&error),"frontier");
                    ids.push_back(plan_string(plan,v.identity));binds.push_back(plan_string(plan,v.binding));
                }
            }
            if(p==0){ii=pi;ib=pib;ei=pe;eb=peb;}
            else if(ii!=pi || ib!=pib || ei!=pe || eb!=peb) throw std::runtime_error("frontiers disagree across shapes");
        }
    }
    auto r=pointers(resident),iip=pointers(ii),ibp=pointers(ib),eip=pointers(ei),ebp=pointers(eb);
    config.resident_tensor_names=r.data();config.resident_tensor_name_count=r.size();
    config.execution_contract=contract.empty()?nullptr:contract.c_str();
    config.activation_import_identities=iip.data();config.activation_import_identity_count=iip.size();config.activation_import_bindings=ibp.data();
    config.activation_export_identities=eip.data();config.activation_export_identity_count=eip.size();config.activation_export_bindings=ebp.data();
    skippy_model * model=nullptr;check(skippy_model_open(path,&config,&model,&error),"open model");
    if(plan)skippy_stage_plan_free(plan);
    if(model->model->hparams.causal_attn || model->ctx->get_memory()!=nullptr) throw std::runtime_error("probe requires stateless noncausal model");
    return model;
}
static std::vector<char> read_payload(size_t size) {
    if(size>MAX_FRAME)throw std::runtime_error("payload exceeds bound");
    std::vector<char> bytes(size);std::cin.read(bytes.data(),size);
    if(!std::cin)throw std::runtime_error("truncated payload");return bytes;
}
static void respond(const json & value,const std::vector<char> & bytes={}) {
    std::cout<<value.dump()<<'\n';if(!bytes.empty())std::cout.write(bytes.data(),bytes.size());std::cout.flush();
    if(!std::cout)throw std::runtime_error("output closed");
}
static json evaluate(skippy_session * session,const std::string & mode,const json & req,std::vector<char> & output) {
    auto tokens=req.at("tokens").get<std::vector<llama_token>>();
    if(tokens.empty() || tokens.size()>MAX_TOKENS)throw std::runtime_error("invalid canvas size");
    auto * vocab=llama_model_get_vocab(session->stage_model->model);int nv=llama_vocab_n_tokens(vocab);
    for(auto token:tokens)if(token<0 || token>=nv)throw std::runtime_error("invalid token id");
    size_t start=req.value("read_start",size_t(0));if(start>tokens.size())throw std::runtime_error("invalid read start");
    const std::string codec=req.value("codec",std::string("f32"));
    if(codec!="f32" && codec!="f16")throw std::runtime_error("unknown lab codec");
    auto incoming=read_payload(req.value("payload_bytes",size_t(0)));
    skippy_activation_desc in{},out{};const void * in_data=nullptr;
    std::vector<float> restored;
    if(mode=="tail") {
        if(incoming.size()<sizeof(in))throw std::runtime_error("missing descriptor");
        std::memcpy(&in,incoming.data(),sizeof(in));
        if(in.payload_bytes>MAX_FRAME || in.payload_bytes%4!=0)throw std::runtime_error("invalid frame size");
        if(codec=="f16") {
            if(in.part_count>SKIPPY_ACTIVATION_MAX_PARTS || in.payload_bytes/2!=incoming.size()-sizeof(in))throw std::runtime_error("f16 length mismatch");
            for(size_t i=0;i<in.part_count;++i)if(in.parts[i].ggml_type!=GGML_TYPE_F32)throw std::runtime_error("f16 lab codec requires F32 parts");
            restored.resize(in.payload_bytes/4);
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(incoming.data()+sizeof(in)),restored.data(),restored.size());
            for(float f:restored)if(!std::isfinite(f))throw std::runtime_error("nonfinite restored activation");
            in_data=restored.data();
        } else {
            if(in.payload_bytes!=incoming.size()-sizeof(in))throw std::runtime_error("frame length mismatch");
            in_data=incoming.data()+sizeof(in);
        }
    } else if(!incoming.empty())throw std::runtime_error("unexpected frame");
    check(skippy_validate_frame_input(session,mode=="tail"?&in:nullptr,in_data,tokens.size(),&error),"frame validation");
    auto inputs=skippy_graph_build_inputs_from_config(&session->stage_model->config);
    skippy_graph_build_inputs_set_activation_tokens(inputs,tokens.data(),tokens.size());
    if(mode=="tail")skippy_graph_build_inputs_set_activation_frontier(inputs,session->stage_model->input_activation_boundary,&in,in_data,0);
    auto batch=llama_batch_init(tokens.size(),0,1);batch.n_tokens=tokens.size();
    for(int i=0;i<batch.n_tokens;++i) {
        batch.token[i]=tokens[i];batch.pos[i]=i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=session->seq_id;batch.logits[i]=mode=="head" || static_cast<size_t>(i)>=start;
    }
    const auto began=std::chrono::steady_clock::now();
    // Fixed positions and no causal history/sampler: each pass replaces the whole canvas.
    check(skippy_session_begin_external_decode(session,&error),"begin canvas");
    int rc=session->ctx->decode(batch,inputs);
    check(skippy_session_end_external_decode(session,&error),"end canvas");llama_batch_free(batch);
    if(rc)throw std::runtime_error("canvas decode failed: "+std::to_string(rc));
    llama_synchronize(session->ctx);
    const double compute=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
    json reply={{"id",req.at("id")},{"positions",tokens.size()},{"compute_s",compute},{"payload_bytes",0}};
    if(mode=="head") {
        std::vector<char> payload(MAX_FRAME);size_t bytes=0;
        check(skippy_prepare_output_activation_frame(session,tokens.size(),payload.data(),payload.size(),&bytes,&out,nullptr,&error),"prepare frame");
        check(skippy_copy_output_activation_frame(session,tokens.size(),payload.data(),nullptr,nullptr,&bytes,&out,&error),"copy frame");
        if(codec=="f16") {
            for(size_t i=0;i<out.part_count;++i)if(out.parts[i].ggml_type!=GGML_TYPE_F32)throw std::runtime_error("f16 lab codec requires F32 parts");
            if(bytes%4!=0)throw std::runtime_error("unaligned F32 payload");
            output.resize(sizeof(out)+bytes/2);std::memcpy(output.data(),&out,sizeof(out));
            ggml_fp32_to_fp16_row(reinterpret_cast<const float *>(payload.data()),reinterpret_cast<ggml_fp16_t *>(output.data()+sizeof(out)),bytes/4);
        } else {
            output.resize(sizeof(out)+bytes);std::memcpy(output.data(),&out,sizeof(out));std::memcpy(output.data()+sizeof(out),payload.data(),bytes);
        }
        reply["payload_bytes"]=output.size();reply["activation_bytes"]=bytes;reply["codec"]=codec;
    } else {
        std::vector<int> ids;std::vector<double> probabilities,top_logits;
        const auto read_began=std::chrono::steady_clock::now();
        for(size_t i=start;i<tokens.size();++i) {
            const float * row=llama_get_logits_ith(session->ctx,i);if(!row)throw std::runtime_error("missing canvas row");
            int best=0;for(int j=0;j<nv;++j){if(!std::isfinite(row[j]))throw std::runtime_error("nonfinite logits");if(row[j]>row[best])best=j;}
            double denominator=0;for(int j=0;j<nv;++j)denominator+=std::exp(double(row[j])-row[best]);
            ids.push_back(best);probabilities.push_back(1/denominator);top_logits.push_back(row[best]);
        }
        reply["ids"]=ids;reply["confidence"]=probabilities;reply["top_logits"]=top_logits;
        reply["readout_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-read_began).count();
    }
    return reply;
}
static void quiet_log(ggml_log_level level,const char * text,void *) {if(level>=GGML_LOG_LEVEL_ERROR)std::fputs(text,stderr);}
int main(int argc,char ** argv) {
    try {
        if(argc!=3)throw std::runtime_error("usage: canvas_worker model full|head|tail");
        const std::string mode=argv[2];if(mode!="full" && mode!="head" && mode!="tail")throw std::runtime_error("unknown mode");
        llama_log_set(quiet_log,nullptr);ggml_log_set(quiet_log,nullptr);llama_backend_init();
        auto * model=open_model(argv[1],mode);skippy_session * session=nullptr;
        check(skippy_session_create(model,&session,&error),"session");auto * vocab=llama_model_get_vocab(model->model);
        respond({{"ready",true},{"mode",mode},{"protocol","trusted-lab-canvas-v1"},{"descriptor_bytes",sizeof(skippy_activation_desc)}});
        std::string line;
        while(std::getline(std::cin,line)) {
            if(line.size()>1024*1024)throw std::runtime_error("request too large");auto req=json::parse(line);std::string op=req.at("op");
            if(op=="stop")break;
            if(op=="tokenize") {
                auto text=req.at("text").get<std::string>();std::vector<llama_token> tokens(MAX_TOKENS);
                int n=llama_tokenize(vocab,text.data(),text.size(),tokens.data(),tokens.size(),false,true);
                if(n<0)throw std::runtime_error("prompt too long");tokens.resize(n);respond({{"tokens",tokens}});
            } else if(op=="detokenize") {
                auto tokens=req.at("tokens").get<std::vector<llama_token>>();if(tokens.size()>MAX_TOKENS)throw std::runtime_error("too many tokens");
                std::string text;
                for(auto t:tokens){char part[1024];int n=llama_token_to_piece(vocab,t,part,sizeof(part),0,true);if(n<0)throw std::runtime_error("token piece too long");text.append(part,n);}
                respond({{"text",text}});
            } else if(op=="eval") {std::vector<char> bytes;auto reply=evaluate(session,mode,req,bytes);respond(reply,bytes);}
            else throw std::runtime_error("unknown operation");
        }
        check(skippy_session_free(session,&error),"free session");check(skippy_model_free(model,&error),"free model");llama_backend_free();return 0;
    } catch(const std::exception & e) {std::fprintf(stderr,"CANVAS_FAILURE: %s\n",e.what());return 2;}
}
