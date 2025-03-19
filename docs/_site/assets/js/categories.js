const categories = { coding: [{ url: `/coding/2023/12/11/diffusion-model-demo2.html`, date: `11 Dec 2023`, title: `A Diffusion Model from Scratch in Pytorch`},{ url: `/coding/2023/12/11/diffusion-model-demo1.html`, date: `11 Dec 2023`, title: `Diffusion Models Tutorial`},{ url: `/coding/2023/12/02/transformer-implementation.html`, date: `02 Dec 2023`, title: `transformer implementation`},{ url: `/coding/2023/12/02/Inspect-BERT-Vocabulary.html`, date: `02 Dec 2023`, title: `Inspect BERT Vocabulary`},{ url: `/coding/2023/12/01/convolutional-neural-networks-step-by-step.html`, date: `01 Dec 2023`, title: `Convolutional Neural Networks: Step by Step`},{ url: `/coding/2023/12/01/Convolution-model-Application-v1a.html`, date: `01 Dec 2023`, title: `Convolutional Neural Networks: Application`},{ url: `/coding/2023/12/01/BERT-Pytorch-Scratch.html`, date: `01 Dec 2023`, title: `BERT Pytorch from Scratch`},{ url: `/coding/2023/11/26/Transformer-learning-C5W4A1SubclassV1.html`, date: `26 Nov 2023`, title: `Transformer Network`},{ url: `/coding/2023/11/02/xgboost-from-scratch.html`, date: `02 Nov 2023`, title: `XGBoost from scratch`},{ url: `/coding/2023/11/01/Understanding-the-performance-of-LightGBM-and-XGBoost.html`, date: `01 Nov 2023`, title: `Understanding the performance of LightGBM and XGBoost`},{ url: `/coding/2023/10/26/BERT-FineTuning-Sentence-Classification-v4.html`, date: `26 Oct 2023`, title: `BERT Fine-Tuning Tutorial with PyTorch`},{ url: `/coding/2023/10/11/expectation-maximization-algorithm.html`, date: `11 Oct 2023`, title: `Expection Maximization`},{ url: `/coding/2023/01/26/neural-network-bach-normalization-scratch.html`, date: `26 Jan 2023`, title: `neural network + Bach normalization`},{ url: `/coding/2023/01/26/Langevin-Monte-Carlo.html`, date: `26 Jan 2023`, title: `Langevin Monte Carlo`},{ url: `/coding/2023/01/26/gbm-implementation.html`, date: `26 Jan 2023`, title: `gbm Implementation`},{ url: `/coding/2023/01/26/GMM-from-scratch.html`, date: `26 Jan 2023`, title: `Gaussian Mixture Model Clearly Explained`},{ url: `/coding/2023/01/26/regression-tree.html`, date: `26 Jan 2023`, title: `Regression Tree Practise`},{ url: `/coding/2023/01/26/gprDemoMarglik.html`, date: `26 Jan 2023`, title: `a Gaussian Process Regression with multiple local minima`},{ url: `/coding/2023/01/26/Bayesian-Learning.html`, date: `26 Jan 2023`, title: `Bayesian Learning`},{ url: `/coding/learning/2023/01/02/GPR.html`, date: `02 Jan 2023`, title: `GPR`},{ url: `/coding/2021/01/26/Bayesian-Optimization-2.html`, date: `26 Jan 2021`, title: `Bayesian Optimization 2`},{ url: `/coding/2021/01/26/Bayesian-Optimization-1.html`, date: `26 Jan 2021`, title: `Bayesian Optimization with GPyOpt`},{ url: `/coding/2021/01/26/mcmc-introduction-3.html`, date: `26 Jan 2021`, title: `Mcmc Introduction 3`},{ url: `/coding/2021/01/26/mcmc-introduction-2.html`, date: `26 Jan 2021`, title: `Mcmc Introduction 2`},],LEARNING: [{ url: `/learning/2024/04/28/Vision-Transformer.html`, date: `28 Apr 2024`, title: `Vision Transformer`},{ url: `/learning/2024/03/25/Mathematics-in-MLSVD-(Singular-value-decomposition)-and-PCA.html`, date: `25 Mar 2024`, title: `Understanding SVD and PCA in Mathematics`},{ url: `/learning/2023/12/01/BERT-GPT-Diffusion-Research.html`, date: `01 Dec 2023`, title: `BERT GPT Diffusion Research`},{ url: `/learning/2023/01/26/RNN-and-LSMT.html`, date: `26 Jan 2023`, title: `RNN and LSMT`},{ url: `/learning/2023/01/26/Gradient-boosting.html`, date: `26 Jan 2023`, title: `Gradient boosting`},{ url: `/coding/learning/2023/01/02/GPR.html`, date: `02 Jan 2023`, title: `GPR`},{ url: `/learning/2021/01/26/Markov-Chain-Monte-Carlo.html`, date: `26 Jan 2021`, title: `Markov Chain Monte Carlo`},{ url: `/learning/2021/01/26/Machine-LearningNote-Pub.html`, date: `26 Jan 2021`, title: `Machine Learning Notes`},],working: [{ url: `/working/2024/03/25/My-Tasks-and-Notes.html`, date: `25 Mar 2024`, title: `working todo`},{ url: `/working/2024/03/01/My-Tasks-and-Notes.html`, date: `01 Mar 2024`, title: `working todo`},{ url: `/working/2023/12/01/My-Tasks-and-Notes.html`, date: `01 Dec 2023`, title: `working todo`},],Project: [{ url: `/project/2024/04/28/Local-RAG-LLM-Project.html`, date: `28 Apr 2024`, title: `Local RAG LLM Project`},], }

console.log(categories)

window.onload = function () {
  document.querySelectorAll(".category").forEach((category) => {
    category.addEventListener("click", function (e) {
      const posts = categories[e.target.innerText.replace(" ","_")];
      let html = ``
      posts.forEach(post=>{
        html += `
        <a class="modal-article" href="${post.url}">
          <h4>${post.title}</h4>
          <small class="modal-article-date">${post.date}</small>
        </a>
        `
      })
      document.querySelector("#category-modal-title").innerText = e.target.innerText;
      document.querySelector("#category-modal-content").innerHTML = html;
      document.querySelector("#category-modal-bg").classList.toggle("open");
      document.querySelector("#category-modal").classList.toggle("open");
    });
  });

  document.querySelector("#category-modal-bg").addEventListener("click", function(){
    document.querySelector("#category-modal-title").innerText = "";
    document.querySelector("#category-modal-content").innerHTML = "";
    document.querySelector("#category-modal-bg").classList.toggle("open");
    document.querySelector("#category-modal").classList.toggle("open");
  })
};