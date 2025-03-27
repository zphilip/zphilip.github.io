---
layout: page
title: CV
permalink: /cv/
---
# Education
Graduated from Zhejiang University <img src="/assets/zu-logo.png" style="display: inline-block; margin: 0; zoom: 4%;" /> at 1996, Bachelor of Computer Science. 

- Age: 49
- Look for AI, Machining Learning related work
- City：Hangzhou, Shanghai

# Experience
Many years Architecture and Speciﬁﬁcation work in Nokia 3G WCDMA RNC, 4G and 5G base station, in different area. My latest postion is : senior system speciﬁﬁcation engineer of the Nokia MN （移动事业部） Architecture and Speciﬁﬁcation.

### 1, (2020 - 2023) 5G base station security area include security transport protocol like TLS/IPSec/SSH, BS Security arArchitecture, BS secure booting and the Architecture and Speciﬁﬁcation work of the 3GPP and O-RAN security standard implemention etc.
    1. Netconf over TLS and SSH. 
        provides the support of Netconf over TLS and FTPES based on X509 v3 operator certification for Fronthaul M-plane interface for O-RU. From security perspective, it complies to the ORAN FH M-Plane speciﬁﬁcation v07 version.
    2. TLS support for Trace, RTPM, PCMD and RFSC collectors.
        This project support Transport Layer Security (TLS) for protecting traffic for trace type interfaces
    3. RU Front Haul TLS X509 manual certification enrollment:
        - Manual Enrollment of operator X.509 certification in the RU (CSR export, cert import)  to be used with IPSEC on the
        fronthaul interface for Classical and Cloud.
        - Creation of the RSA  key pair and of the CSR with export.
        - Import of the signed certification and trust chain.  
        - Mandates a local tool and  connectivity to the RU for manual enrollment management
    4. BTS Secondary IPsec backup tunnel over satellite
        - Project content: project supports the configuration of two alternative IPsec tunnels for same traffic, only one being established and carrying traffic at a time, to allow i.e. geo-redundant Security Gateway (SEG) deployments for SEG site disaster recovery.
        - include adding extra IPSec tunnel via satellite, adding extra traffic based admission controlling and adding extra cplane/cell controlling
        - Role: Project Architecture and leader of the Project
    5. IPSec supporting in cloud BTS (4G)
        The project provides the transport IPsec feature set for the LTE speciﬁﬁc functionality in Cloud BTS
        - The standard IPsec considerations for the S1 and X2 interfaces on the backhaul of the veNB.
        - IPsec on the new Fronthaul interface C1 between the RAP’s and the veNB and C2 ( between RAP’s)
        this project supports the configuration of two alternative IPsec tunnels for same traffic, only one being established and carrying traffic at a time, to allow i.e. geo-redundant Security Gateway (SEG) deployments for SEG site disaster recovery.

### 2, (2017 - 2023) AI machine learning 4G and 5G internal improvement projeccts (python, tensorﬂﬂow, pytorch):

    1. A Machine Learning Based Sync Input Detection for BTS
        By introduce extra Machine learning based abnormal detect to trigger the hold procedure in time so that the system could much more better holdover performance, at same time alarm the customer that the sync input reference is invalid and holdover happens.
    2. Detection and prevention of DDOS with AIML for BTS
        Detecting the pre-deﬁﬁned Intrusion attack to BTS (DDOS only in phase1) based on AI/ML solution (ready AI/ML
        solution from Bell Lab) so that BTS could take pre-deﬁﬁned actions to prevent BTS service down while Intrusion is ongoing and recovery BTS service in time while Intrusion is end.
    3. A Machine learning based Transport admission control for BTS 
        This project study a new machine learning method: GPR or Sparse GPR based TAC. The GPR based TAC has ability to learn real traffic load distribution according to multiple features such as the real time current traffic load, the call maximum bit rate and the accepted admitted bandwidth since last epoch time, which are used in old MBTAC. In addition, features that including call holding time, call number etc. are utilised to enhance the performance of the proposed approach.
    4. A compute vision aidded indoor mobile location
        - Collecting the UE mobility radio data via the own developed android software
        - Collecting the Camera data via the normal monitoring camera
        This Project introduces one machine leaning based UE indoor location service by combining the UE radio tracking information from the connected BTS and object tracking information from the cooperating camera. The UE radio tracking information from BTS is the radio signal information like RSSI, RSRP, RSRQ, CQI, SINR and TA etc., which is measured from the related serving cell or neighbor cells. Diﬀﬀerent telecom generation like 4G and 5G may have different UE radio information, for example 5G beam related measurement. Multiple object tracking system (MOTS) with the cooperating cameras initial the objects tracking in real time and retrieve the multiple objects tracking information (OTI) with the relative map or real-world location information/coordination which is transferred from the camera image point coordination.          

### 3, (2016 - 2020) Nokia bare-metal BTS and cloud BTS transport Architecture engineer(传输架构工程师). The work include base station internal transport/network (L2/L3) architecture between BTS RU(无线单元) and BTS Could VNF(云基站), the fast path (云基站内部快速通道) for user traffic and slow path (慢速通道) for other traffic in BTS Cloud VNF,  and BTS synchronization/timing.

    1. Cloud BTS VRAN 1.0/2.0 Transport Architecture work
        - Cloud BTS transport/synchronization architecture work, for example MBTAC (measure based TAC) for RAP/cBTS VNF architecture, ICOM gateway for cloud BTS (internal communciation architecture), cBTS transport capacity/performance envulation work, RAP/Cloud BTS transport recommended conﬁﬁguration etc.
        - Cloud BTS VRAN 2.0 transport architecture FSY work.
    2. Cloud BTS SR IOV based Virtual Ethernet Interface
        provide RAN level requirements for SR-IOV based Ethernet interface.
        This feature is to support Single Root I/O Virtualization (SR-IOV) based vNIC in Cloud BTS xxx to meet requirements from VCP Edge (Verizon Cloud Platform name, aka Corona in earlier phase). It shall adhere to standard SRIOV function for ensuring interoperability based on PCI-SIG SRIOV specification
    3. Cloud BTS IPv4/IPv6 based ICOM Gateway （between Radio AP and eNB cloud）
        CBTS splits the eNB into one component running in the cloud and another component, the radio access point (RAP). The communication within the eNB between these components is based on Nokia propritary internal protocols. The new deployment requires these protocols to traverse IPv4 / IPv6 based networks.
        Here comes the ICOM GW into the picture. This entity tunnels or interworks the internal protocols across IPv4 / IPv6 based networks, allowing the cloud based components to communicate with the RAPs. The ICOM GW further supports beside QoS and security aspects the whole feature set TRS provides. Since for UP different internal protocols are utilized per RAP HW type, the ICOM GW could also be understood as a kind of protocol harmonizer.
    4. Cloud BTS VCP Edge VM deployment and Dimensioning
        This project provide :
        a. capacity throughput of Cloud BTS VNF considering :
        - traffic model
        - Computational node
        - Scalability Support (Scale Up/Down not supported)
        b. support VM deployment on xxx Platform
        c. follow the same architecture split as proposed in another project.
    5， BTS Auto detection and adjustment TDD
        Find way to detect BTS Timinging problem. With utilizing the SSTD measurement in LTE dual connectivity and the TA measurement to detect and determine the TimingOffset between an E-UTRA eNB with PCell and an E-UTRA eNB with PSCell (for LTE-DC) Or utilizing the SFTD measurement in 5G and the TA measurement to detect and determine the TimingOffset between:
        - an E-UTRA eNB with PCell and an NR with PSCell (for EN-DC),
        - an NR with PCell and an E-UTRA eNB with PSCell (for NE-DC),
        - an NR with PCell and an NR with PSCell (for NR-DC)
        - an E-UTRA eNB/gNB and an NR with neighbor cell.
        UE or BTS may do the TimingOffset calculation based on the SSTD/SFTD and TA information of the pair BTSes above. Based on the calculated TimingOffset from multiple BTS pairs, the Timing Alignment Error of the related BTS may be determined. BTS calculate the AlignOffset which is the adjustment time shift for next step based on TimingOffset and the multiple actions could be taken to correct such timing alignment error include TDD UL-DL frame structure and special subframe pattern reconfiguration and the BTS internal timing offset adjustment accordingly.

### 4, (2005 - 2015) WCDMA RNC high availability engineer (系统高可靠性架构工程师) and system upgrade-ability domain leader.
    1. Leading RNC system upgrade-ability Architect and speciﬁﬁcation work.
    2. RNC Capacity and Performance Architect and speciﬁﬁcation

### 5, (2003 — 2005) WCDMA RNC high availability recovery system senior software engineer. 
    1. Recovery Archecture design. 
    2. Recovery core software implementation in RNC IPA platform

### 6, 2001-2003 I have about 1.5 years work on Shanghai Bell-Actel as software engineer, major working on the network mangement (SNMP etc) of one L2/L3 switch.

### 7, 2000-2001 I have about 1 years work on alibaba.com (about 100 person), I am java engineer for alibaba.com web site (btw I have SUN java certification at that time). Also I build the intranet for alibaba.com.

### Related marks:
    * Winner of the Nokia hangzhou 2019 Future-X Innovation Incubation program
    * Top 10 inventors in nokia hangzhou for patent ﬁﬁling in 2018
    * Nokia internal Patent:
        - NC325570, a new method of auto detection and adjustment TDD air interface timing oﬀﬀset
        - NC324326, a machine learning based method for sync input degradation detection and holdover.
        - NC319887, AN IPSEC SPI BASED TRAFFIC FORWARD METHOD
        - NC317851, AN IPSEC IKE PROTOCOL SUPPLEMENT FOR DYNAMIC BYPASS NEGOTIATION"
        - NC306972, A S1/X2 INTERFACE DOWNLINK GTP-U TEID GENERATION AND FAST DATA FORWARDING METHOD. etc
        - NC326121, A COMPUTE VISION AIDDED INDOOR MOBILE LOCATIONMETHOD


